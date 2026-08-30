# ============================================================
# Cell 0 — Cài đặt (chạy 1 lần đầu nếu cần)
# ============================================================
# !pip install -q sentence-transformers

# ============================================================
# Cell 1 — Imports & đường dẫn data
# ============================================================
import gc, json, time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATASET_SLUG   = "uit-legalir-corpus"   # slug dataset Kaggle của bạn
CHUNKS_PATH    = f"/kaggle/input/{DATASET_SLUG}/chunks_corpus.jsonl"
QUESTIONS_PATH = f"/kaggle/input/{DATASET_SLUG}/val_preprocessed.jsonl"
OUT_DIR        = Path("/kaggle/working")
TOP_N          = 200

# Kiểm tra GPU
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

# ============================================================
# Cell 2 — Load data (1 lần dùng chung cho cả 2 model)
# ============================================================
def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"  {len(chunks):,} chunks")
    return chunks

def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                questions.append({
                    "question_id": str(obj["question_id"]),
                    "question": obj["question"],
                })
    print(f"  {len(questions):,} questions")
    return questions

print("Loading data...")
chunks    = load_chunks(CHUNKS_PATH)
questions = load_questions(QUESTIONS_PATH)
chunk_doc_ids = [str(c["doc_id"]) for c in chunks]

# ============================================================
# Cell 3 — Helper functions (dùng chung)
# ============================================================
def top_n_indices(scores, top_n):
    top_n = min(top_n, len(scores))
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return idx[np.argsort(-scores[idx])]

def encode_and_save_ranks(
    model_name, rank_out_name, *,
    query_prefix="", passage_prefix="",
    query_instruct=None, query_prompt=None,
    max_seq_length=512, trust_remote_code=False,
    batch_size_corpus=64, batch_size_query=64,
):
    """Encode corpus + queries, lưu ranks_*.npz. Tự giải phóng VRAM sau khi xong."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus_cache = OUT_DIR / f"corpus_{rank_out_name}.npz"
    ranks_path   = OUT_DIR / f"{rank_out_name}.npz"

    print(f"\n{'='*60}")
    print(f"▶ {rank_out_name}  ({model_name})")
    print(f"{'='*60}")
    t_start = time.time()

    # --- Load model ---
    print(f"Loading model...")
    model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)
    model.max_seq_length = max_seq_length
    if device == "cuda":
        model = model.half()
    print(f"  model.prompts = {getattr(model, 'prompts', {})}")

    # --- Encode corpus ---
    if corpus_cache.exists():
        print(f"Corpus cache found — loading...")
        chunk_embeddings = np.load(corpus_cache)["embeddings"]
        print(f"  shape: {chunk_embeddings.shape}")
    else:
        print(f"Encoding {len(chunks):,} corpus chunks (batch={batch_size_corpus})...")
        enc_kw = dict(batch_size=batch_size_corpus, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=True)
        texts = [passage_prefix + c["passage"] for c in chunks]
        chunk_embeddings = model.encode(texts, **enc_kw).astype(np.float32)
        elapsed = (time.time() - t_start) / 60
        print(f"  Done  ({elapsed:.1f} min), shape={chunk_embeddings.shape}")
        np.savez_compressed(str(corpus_cache),
                            embeddings=chunk_embeddings,
                            doc_ids=np.array(chunk_doc_ids, dtype=object))
        print(f"  Saved corpus cache: {corpus_cache}")

    # --- Encode queries ---
    print(f"Encoding {len(questions)} queries...")
    t_q = time.time()
    all_ranks = np.zeros((len(questions), TOP_N), dtype=np.int32)
    qids = []

    for start in range(0, len(questions), batch_size_query):
        batch = questions[start: start + batch_size_query]
        enc_kw_q = dict(batch_size=batch_size_query, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
        if query_prompt:
            enc_kw_q["prompt_name"] = query_prompt
            batch_texts = [q["question"] for q in batch]
        elif query_instruct:
            batch_texts = [f"Instruct: {query_instruct}\nQuery: " + q["question"] for q in batch]
        else:
            batch_texts = [query_prefix + q["question"] for q in batch]

        emb  = model.encode(batch_texts, **enc_kw_q).astype(np.float32)
        sims = emb @ chunk_embeddings.T
        for i, q in enumerate(batch):
            all_ranks[start + i] = top_n_indices(sims[i], TOP_N)
            qids.append(str(q["question_id"]))

        done = min(start + batch_size_query, len(questions))
        if done % 200 == 0 or done == len(questions):
            print(f"  {done}/{len(questions)} queries  ({time.time()-t_q:.0f}s)")

    print(f"Query encoding done ({(time.time()-t_q):.1f}s)")

    # --- Lưu ranks ---
    np.savez_compressed(
        str(ranks_path),
        question_ids=np.array([str(q) for q in qids]),
        ranks=all_ranks.astype(np.int32),
    )
    total_min = (time.time() - t_start) / 60
    print(f"\n✅ Saved: {ranks_path}  shape={all_ranks.shape}  total={total_min:.1f} min")

    # --- Giải phóng VRAM trước khi load model tiếp ---
    del model, chunk_embeddings, emb, sims
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"VRAM freed. Ready for next model.")
    return ranks_path

# ============================================================
# Cell 4 — Encode v1  (AITeamVN/Vietnamese_Embedding)
# ~1h on T4, encoder model
# ============================================================
encode_and_save_ranks(
    model_name        = "AITeamVN/Vietnamese_Embedding",
    rank_out_name     = "ranks_v1",
    query_prefix      = "",    # symmetric model — NO prefix
    passage_prefix    = "",
    query_instruct    = None,
    query_prompt      = None,
    max_seq_length    = 2048,  # theo model card AITeamVN
    trust_remote_code = False,
    batch_size_corpus = 128,   # encoder → batch lớn hơn được
    batch_size_query  = 128,
)

# ============================================================
# Cell 5 — Encode harrier  (mainguyen9/vietlegal-harrier-0.6b)
# ~5-6h on T4, decoder model — chạy sau v1 tự động
# ============================================================
encode_and_save_ranks(
    model_name        = "mainguyen9/vietlegal-harrier-0.6b",
    rank_out_name     = "ranks_harrier",
    query_prefix      = "",
    passage_prefix    = "",
    query_instruct    = "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question",
    query_prompt      = None,   # dùng instruct text thay vì prompt_name
    max_seq_length    = 1024,
    trust_remote_code = True,   # ⚠️ harrier yêu cầu trust_remote_code
    batch_size_corpus = 32,
    batch_size_query  = 64,
)

# ============================================================
# Cell 6 — Tổng kết
# ============================================================
print("\n" + "="*60)
print("✅ Hoàn tất! Download 2 file sau từ tab Output:")
for name in ["ranks_v1.npz", "ranks_harrier.npz"]:
    p = OUT_DIR / name
    if p.exists():
        print(f"   {name}  ({p.stat().st_size/1024:.0f} KB)")
    else:
        print(f"   {name}  ← MISSING, kiểm tra lại lỗi bên trên")
print("\nSau đó đặt vào outputs/cache/ trong repo local rồi chạy fuse sweep.")
