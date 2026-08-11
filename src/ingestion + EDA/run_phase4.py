import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.paths import PROCESSED_DATA_DIR
from src.config import chunking_config
import chunker

def process_doc_for_chunking(doc):
    # Determine chunk limits from config, assuming 1 token ≈ 1.3 words.
    # 512 tokens ≈ 350 words roughly.
    max_words = int(chunking_config.MAX_SEQ_LENGTH / 1.4)
    overlap = int(chunking_config.OVERLAP_TOKENS / 1.4)
    
    return chunker.chunk_document(doc, max_words=max_words, overlap=overlap)

def run_phase4():
    print("=== STARTING PHASE 4: CHUNKING & INDEXING PREPARATION ===")
    
    input_file = PROCESSED_DATA_DIR / 'corpus_preprocessed.jsonl'
    output_file = PROCESSED_DATA_DIR / 'chunks_corpus.jsonl'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run phase 2 first.")
        return
        
    print(f"Loading corpus from {input_file}...")
    corpus = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))
            
    print(f"Loaded {len(corpus)} documents. Starting multi-threaded chunking...")
    
    all_chunks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_doc_for_chunking, corpus), total=len(corpus), desc="Chunking Corpus"))
        
    for doc_chunks in results:
        all_chunks.extend(doc_chunks)
        
    print(f"\nChunking complete. Total chunks generated: {len(all_chunks)}")
    
    print(f"Saving chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as out_fh:
        for chunk in all_chunks:
            out_fh.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    # Simple validation stats
    print("\n--- VALIDATION & STATS ---")
    chunk_lengths = [chunker.count_words(c['passage']) for c in all_chunks]
    max_len = max(chunk_lengths) if chunk_lengths else 0
    avg_len = sum(chunk_lengths)/len(chunk_lengths) if chunk_lengths else 0
    print(f"Max word count per chunk: {max_len}")
    print(f"Average word count per chunk: {avg_len:.1f}")
    
    print("=== PHASE 4 COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    run_phase4()
