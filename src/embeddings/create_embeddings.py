import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def process_and_embed():
    input_file = os.path.join("data", "processed", "chunks_corpus.jsonl")
    model_name = "bqbbao6/VN_legal_embedding_512"    
    # Tạo các thư mục lưu trữ
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)
    
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    device=device
    )
    dimension = model.get_sentence_embedding_dimension()
    
    # Khởi tạo FAISS index với Inner Product (Cosine Similarity sau khi normalize)
    index = faiss.IndexFlatIP(dimension)
    
    metadata_file = os.path.join("embeddings", "metadata.jsonl")
    embeddings_file = os.path.join("embeddings", "corpus_embeddings.npy")
    vectorstore_file = os.path.join("vectorstore", "corpus.index")
    
    batch_size = 2048
    batch_texts = []
    
    all_embeddings = []
    
    # Đếm số dòng để hiển thị progress bar
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print("Generating embeddings...")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(metadata_file, 'w', encoding='utf-8') as f_meta:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing and Embedding"):
            data = json.loads(line)
            # Ưu tiên sử dụng đoạn văn bản đã được word segmentation
            text = data.get("passage", "")
            text = "passage: " + text
            batch_texts.append(text)
            
            # Lưu lại metadata
            meta = {
                "doc_id": data.get("doc_id"),
                "chunk_id": data.get("chunk_id"),
                "legal_section": data.get("legal_section"),
                "passage": data.get("passage")
            }
            f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
            
            # Xử lý theo batch
            if len(batch_texts) >= batch_size:
                # Tạo embedding và normalize để dùng với IndexFlatIP (tương đương cosine)
                emb = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
                all_embeddings.append(emb)
                index.add(emb)
                
                # Reset batch
                batch_texts = []
                
        # Xử lý phần còn lại chưa đủ batch
        if len(batch_texts) > 0:
            emb = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(emb)
            index.add(emb)
            
    # Lưu embeddings nguyên bản
    print("Saving embeddings array...")
    final_embeddings = np.vstack(all_embeddings)
    np.save(embeddings_file, final_embeddings)
    
    # Lưu FAISS index vào thư mục vectorstore
    print(f"Saving FAISS index to {vectorstore_file}...")
    faiss.write_index(index, vectorstore_file)
    
    print("Embedding process completed successfully.")
    print(f"- Metadata saved at: {metadata_file}")
    print(f"- Embeddings saved at: {embeddings_file}")
    print(f"- Vectorstore saved at: {vectorstore_file}")

if __name__ == "__main__":
    process_and_embed()
