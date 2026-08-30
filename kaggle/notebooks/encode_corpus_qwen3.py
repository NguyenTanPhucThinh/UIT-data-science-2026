# ============================================================
# Cell 0 — Cài đặt thư viện (chạy 1 lần đầu)
# ============================================================
# !pip install -q sentence-transformers==3.0.1 rank_bm25

# ============================================================
# Cell 1 — Config: THAY ĐỔI Ở ĐÂY nếu muốn dùng model khác
# ============================================================
import os, json, time
from pathlib import Path

# --- Model config ---
MODEL_NAME       = "Qwen/Qwen3-Embedding-0.6B"   # đổi thành "CATI-AI/..." khi access được duyệt
QUERY_INSTRUCT   = "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question"
PASSAGE_PREFIX   = ""          # Qwen3-Embedding dùng prompt_name, không cần prefix thủ công
QUERY_PROMPT     = "query"     # tên prompt trong model.prompts cho query (kiểm tra bằng print(model.prompts))
PASSAGE_PROMPT   = None        # None = không dùng prompt_name cho passage (để trống)
MAX_SEQ_LENGTH   = 1024        # Qwen3-Embedding hỗ trợ đến 32k, 1024 là đủ cho chunk pháp lý
TRUST_REMOTE     = False       # Qwen3-Embedding-0.6B: không cần trust_remote_code
BATCH_SIZE       = 32          # T4: 32 là ổn, giảm xuống 16 nếu OOM

RANK_OUT_NAME    = "ranks_qwen3_emb"  # tên file output (không có .npz)
TOP_N            = 200         # số chunk lưu mỗi câu hỏi (giống cấu hình ensemble_fusion)

# --- Đường dẫn data (Kaggle dataset mount point) ---
# Kaggle mount dataset vào /kaggle/input/<dataset-slug>/
# Tìm đúng tên slug bằng cách chạy: !ls /kaggle/input/
DATASET_SLUG     = "uit-legalir-corpus"   # slug khi tạo dataset
CHUNKS_PATH      = f"/kaggle/input/{DATASET_SLUG}/chunks_corpus.jsonl"
QUESTIONS_PATH   = f"/kaggle/input/{DATASET_SLUG}/val_preprocessed.jsonl"
OUT_DIR          = Path("/kaggle/working")

print(f"Model  : {MODEL_NAME}")
print(f"Chunks : {CHUNKS_PATH}")
print(f"Questions: {QUESTIONS_PATH}")
print(f"Output : {OUT_DIR}")

# ============================================================
# Cell 2 — Kiểm tra GPU
# ============================================================
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# Cell 3 — Load data
# ============================================================
import re
import numpy as np

def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(obj)
    print(f"  Loaded {len(chunks):,} chunks")
    return chunks

def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            questions.append({
                "question_id": str(obj["question_id"]),
                "question": obj["question"],
                "question_segmented": obj.get("question_segmented"),
            })
    print(f"  Loaded {len(questions):,} questions")
    return questions

print("Loading data...")
chunks = load_chunks(CHUNKS_PATH)
questions = load_questions(QUESTIONS_PATH)
chunk_doc_ids = [str(c["doc_id"]) for c in chunks]

# ============================================================
# Cell 4 — Load model
# ============================================================
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {DEVICE}...")

t0 = time.time()
model = SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=TRUST_REMOTE)
model.max_seq_length = MAX_SEQ_LENGTH
if DEVICE == "cuda":
    model = model.half()   # fp16 để tăng tốc
print(f"Model loaded  ({time.time()-t0:.1f}s)")
print(f"model.prompts = {getattr(model, 'prompts', {})}")

# ============================================================
# Cell 5 — Encode corpus (phần TỐN THỜI GIAN NHẤT, ~1-2h)
# ============================================================
CORPUS_CACHE = OUT_DIR / f"corpus_emb_{RANK_OUT_NAME}.npz"

if CORPUS_CACHE.exists():
    print(f"Corpus cache found, loading...")
    data = np.load(CORPUS_CACHE)
    chunk_embeddings = data["embeddings"]
    print(f"  Shape: {chunk_embeddings.shape}")
else:
    print(f"Encoding {len(chunks):,} chunks (this takes ~1-2h on T4)...")
    t0 = time.time()
    encode_kwargs = dict(
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if PASSAGE_PROMPT:
        encode_kwargs["prompt_name"] = PASSAGE_PROMPT
        texts = [c["passage"] for c in chunks]
    else:
        texts = [PASSAGE_PREFIX + c["passage"] for c in chunks]

    chunk_embeddings = model.encode(texts, **encode_kwargs).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Encoding done: {elapsed/60:.1f} min, shape={chunk_embeddings.shape}")

    # Lưu corpus cache
    doc_ids_arr = np.array(chunk_doc_ids, dtype=object)
    np.savez_compressed(str(CORPUS_CACHE), embeddings=chunk_embeddings, doc_ids=doc_ids_arr)
    print(f"Saved corpus cache: {CORPUS_CACHE}")

# ============================================================
# Cell 6 — Encode queries + sinh ranks (vài phút)
# ============================================================
def top_n_indices(scores, top_n):
    n = len(scores)
    top_n = min(top_n, n)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    return idx[np.argsort(-scores[idx])]

print(f"Encoding {len(questions)} queries + computing ranks...")
t0 = time.time()

QUERY_BATCH = 64
all_ranks = np.zeros((len(questions), TOP_N), dtype=np.int32)
qids = []

for start in range(0, len(questions), QUERY_BATCH):
    batch = questions[start: start + QUERY_BATCH]
    encode_kwargs_q = dict(
        batch_size=QUERY_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if QUERY_PROMPT:
        encode_kwargs_q["prompt_name"] = QUERY_PROMPT
        batch_texts = [q["question"] for q in batch]
    else:
        batch_texts = [f"Instruct: {QUERY_INSTRUCT}\nQuery: " + q["question"] for q in batch]

    emb = model.encode(batch_texts, **encode_kwargs_q).astype(np.float32)
    sims = emb @ chunk_embeddings.T

    for i, q in enumerate(batch):
        all_ranks[start + i] = top_n_indices(sims[i], TOP_N)
        qids.append(str(q["question_id"]))

    done = min(start + QUERY_BATCH, len(questions))
    if done % 200 == 0 or done == len(questions):
        print(f"  {done}/{len(questions)} queries ({time.time()-t0:.0f}s)")

print(f"Query encoding done ({time.time()-t0:.1f}s)")

# ============================================================
# Cell 7 — Lưu ranks_*.npz (file này download về local)
# ============================================================
ranks_path = OUT_DIR / f"{RANK_OUT_NAME}.npz"
np.savez_compressed(
    str(ranks_path),
    question_ids=np.array([str(q) for q in qids]),
    ranks=all_ranks.astype(np.int32),
)
print(f"\n{'='*60}")
print(f"✅ Saved: {ranks_path}")
print(f"   Shape: {all_ranks.shape}")
print(f"\n➡️  Download file '{RANK_OUT_NAME}.npz' từ tab Output bên phải")
print(f"   rồi đặt vào outputs/cache/ trong repo local")
print(f"   Sau đó chạy fuse sweep LOCAL (xem kaggle/README.md)")
