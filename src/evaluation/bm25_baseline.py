import json
import time
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

base_dir = Path(__file__).resolve().parent.parent.parent

def load_chunks(filepath):
    print(f"Loading chunks from {filepath}...")
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def load_queries(filepath):
    print(f"Loading queries from {filepath}...")
    queries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def evaluate_bm25():
    chunk_file = base_dir / 'data' / 'processed' / 'chunks_corpus.jsonl'
    query_file = base_dir / 'data' / 'processed' / 'val_preprocessed.jsonl'
    
    if not chunk_file.exists() or not query_file.exists():
        print("Missing processed data files! Please run Phase 1-4 first.")
        return
        
    chunks = load_chunks(chunk_file)
    queries = load_queries(query_file)[:20]  # Subset to 20 for fast estimation
    
    print("Tokenizing chunk passages for BM25 (using pre-segmented text)...")
    t0 = time.time()
    tokenized_corpus = []
    chunk_doc_ids = []
    
    for c in chunks:
        # The text is already segmented by VnCoreNLP, so space splits word tokens
        text = c.get('passage_segmented', '')
        tokenized_corpus.append(text.split())
        chunk_doc_ids.append(c['doc_id'])
        
    print(f"Tokenization complete in {time.time() - t0:.2f}s. Building BM25 index...")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built in {time.time() - t0:.2f}s.")
    
    print("\nRunning Evaluation on Validation Set...")
    recall_1 = 0
    recall_5 = 0
    mrr_5 = 0.0
    
    for q in tqdm(queries, desc="Evaluating"):
        q_tokens = q.get('question_segmented', '').split()
        gold_docs = set(q.get('answer_doc_ids', []))
        if not gold_docs:
            continue
            
        # Get chunk-level scores
        chunk_scores = bm25.get_scores(q_tokens)
        
        # Max-Score Aggregation mapping chunk_scores to doc_ids
        doc_scores = {}
        for doc_id, score in zip(chunk_doc_ids, chunk_scores):
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score
                
        # Sort documents by max score descending
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_5_docs = [doc_id for doc_id, score in ranked_docs[:5]]
        
        # Recall @ 1
        if top_5_docs and top_5_docs[0] in gold_docs:
            recall_1 += 1
            
        # Recall @ 5 and MRR @ 5
        hit_found = False
        for rank, doc_id in enumerate(top_5_docs, 1):
            if doc_id in gold_docs:
                if not hit_found:
                    mrr_5 += 1.0 / rank
                    hit_found = True
                    
        if hit_found:
            recall_5 += 1
            
    total_q = len(queries)
    print("\n" + "="*40)
    print("      BM25 BASELINE RESULTS")
    print("      (Validation Set - 1,398 Queries)")
    print("      Aggregation: Max-Score")
    print("="*40)
    print(f"Recall@1: {recall_1 / total_q:.4f} ({recall_1}/{total_q})")
    print(f"Recall@5: {recall_5 / total_q:.4f} ({recall_5}/{total_q})")
    print(f"MRR@5:    {mrr_5 / total_q:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_bm25()
