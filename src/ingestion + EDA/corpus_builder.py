import os
import json
import re

def infer_doc_type(name: str, link: str, passage: str) -> str:
    """Infer document type from name, link, or text passage."""
    text = (name or '') + ' ' + (link or '') + ' ' + (passage[:500] if passage else '')
    text_upper = text.upper()
    
    if 'NGHỊ ĐỊNH' in text_upper or 'NGHI-DINH' in text_upper or 'ND-CP' in text_upper:
        return 'Nghị định'
    elif 'THÔNG TƯ' in text_upper or 'THONG-TU' in text_upper or 'TT-' in text_upper:
        return 'Thông tư'
    elif 'QUYẾT ĐỊNH' in text_upper or 'QUYET-DINH' in text_upper or 'QD-' in text_upper:
        return 'Quyết định'
    elif 'LUẬT' in text_upper or 'LUAT-' in text_upper:
        return 'Luật'
    elif 'CÔNG VĂN' in text_upper or 'CONG-VAN' in text_upper:
        return 'Công văn'
    elif 'NGHỊ QUYẾT' in text_upper or 'NGHI-QUYET' in text_upper:
        return 'Nghị quyết'
    elif 'CHỈ THỊ' in text_upper or 'CHI-THI' in text_upper:
        return 'Chỉ thị'
    else:
        return 'Khác'

def build_unified_corpus(raw_context_dir: str, output_filepath: str):
    """Reads all context_*.json files and creates a unified corpus.jsonl file."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    files = [f for f in os.listdir(raw_context_dir) if f.startswith('context_') and f.endswith('.json')]
    print(f"Found {len(files)} context files to process.")
    
    records = []
    empty_docs = []
    missing_names = []
    
    for filename in sorted(files):
        filepath = os.path.join(raw_context_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        
        doc_id = str(data.get('id'))
        name = data.get('name')
        link = data.get('link', '')
        passage = data.get('passage', '')
        passage_len = len(passage) if passage else 0
        
        is_empty = (passage_len == 0)
        has_name = name is not None and len(str(name).strip()) > 0
        
        if is_empty:
            empty_docs.append(doc_id)
        if not has_name:
            missing_names.append(doc_id)
            
        doc_type = infer_doc_type(name, link, passage)
        
        record = {
            'doc_id': doc_id,
            'name': name if has_name else None,
            'link': link,
            'passage': passage,
            'passage_length': passage_len,
            'has_name': has_name,
            'is_empty': is_empty,
            'doc_type': doc_type
        }
        records.append(record)
        
    # Write to corpus.jsonl
    with open(output_filepath, 'w', encoding='utf-8') as out_fh:
        for r in records:
            out_fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"Successfully wrote {len(records)} records to {output_filepath}")
    print(f"Empty passages count: {len(empty_docs)}")
    print(f"Missing name count: {len(missing_names)}")
    return records, empty_docs, missing_names

if __name__ == '__main__':
    raw_dir = 'data/raw/selected-contexts'
    out_path = 'data/processed/corpus.jsonl'
    build_unified_corpus(raw_dir, out_path)
