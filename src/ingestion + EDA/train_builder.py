import os
import json
import random

def build_train_and_val(train_filepath: str, public_filepath: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """Processes train.json and public-official.json into clean jsonl files and creates train/val split."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train.json
    with open(train_filepath, 'r', encoding='utf-8') as fh:
        train_data = json.load(fh)
        
    train_records = []
    referenced_doc_ids = set()
    
    for q_id, val in train_data.items():
        question = val['question']
        answers = [str(a) for a in val['answer']] if val.get('answer') else []
        for a in answers:
            referenced_doc_ids.add(a)
            
        train_records.append({
            'question_id': str(q_id),
            'question': question,
            'answer_doc_ids': answers,
            'num_answers': len(answers)
        })
        
    print(f"Total train questions: {len(train_records)}")
    print(f"Total unique referenced document IDs in train: {len(referenced_doc_ids)}")
    
    # Write train_clean.jsonl
    train_clean_path = os.path.join(output_dir, 'train_clean.jsonl')
    with open(train_clean_path, 'w', encoding='utf-8') as out_fh:
        for r in train_records:
            out_fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {train_clean_path}")
    
    # Create Stratified Train/Val Split based on num_answers
    random.seed(seed)
    # Group by num_answers to preserve distribution
    by_num_answers = {}
    for r in train_records:
        n_ans = r['num_answers']
        if n_ans not in by_num_answers:
            by_num_answers[n_ans] = []
        by_num_answers[n_ans].append(r)
        
    train_split = []
    val_split = []
    
    for n_ans, items in by_num_answers.items():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val_split.extend(items[:n_val])
        train_split.extend(items[n_val:])
        
    print(f"Split results -> Train: {len(train_split)}, Val: {len(val_split)}")
    
    val_clean_path = os.path.join(output_dir, 'val_clean.jsonl')
    with open(val_clean_path, 'w', encoding='utf-8') as out_fh:
        for r in val_split:
            out_fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {val_clean_path}")
    
    split_info_path = os.path.join(output_dir, 'train_val_split.json')
    split_info = {
        'train_qids': [r['question_id'] for r in train_split],
        'val_qids': [r['question_id'] for r in val_split],
        'seed': seed,
        'val_ratio': val_ratio
    }
    with open(split_info_path, 'w', encoding='utf-8') as out_fh:
        json.dump(split_info, out_fh, indent=2, ensure_ascii=False)
    print(f"Saved {split_info_path}")
    
    # Process public-official.json
    with open(public_filepath, 'r', encoding='utf-8') as fh:
        public_data = json.load(fh)
        
    public_records = []
    for q_id, val in public_data.items():
        question = val['question']
        public_records.append({
            'question_id': str(q_id),
            'question': question,
            'answer_doc_ids': None
        })
        
    public_clean_path = os.path.join(output_dir, 'public_test.jsonl')
    with open(public_clean_path, 'w', encoding='utf-8') as out_fh:
        for r in public_records:
            out_fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved {public_clean_path}")
    
    return train_records, val_split, public_records, referenced_doc_ids

if __name__ == '__main__':
    train_path = 'data/raw/train.json'
    public_path = 'data/raw/public-official.json'
    out_dir = 'data/processed'
    build_train_and_val(train_path, public_path, out_dir)
