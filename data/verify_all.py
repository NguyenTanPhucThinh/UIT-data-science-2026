import os
import json
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

project_root = Path(__file__).resolve().parent.parent

def audit_all():
    print("==================================================")
    print("       FULL PIPELINE & DATASET INTEGRITY AUDIT    ")
    print("==================================================")
    
    processed_dir = project_root / 'data' / 'processed'
    
    files_to_check = {
        'corpus.jsonl': 8532,
        'corpus_preprocessed.jsonl': 8532,
        'train_clean.jsonl': 7000,
        'train_preprocessed.jsonl': 7000,
        'val_clean.jsonl': 1398,
        'val_preprocessed.jsonl': 1398,
        'public_test.jsonl': 1000,
        'public_test_preprocessed.jsonl': 1000,
    }
    
    all_passed = True
    
    print("\n1. Checking File Existence & Record Counts:")
    for fname, expected_count in files_to_check.items():
        fpath = processed_dir / fname
        if not fpath.exists():
            print(f"  ❌ FAIL: {fname} does not exist!")
            all_passed = False
            continue
            
        # Count lines
        count = 0
        with open(fpath, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    count += 1
                    
        if count == expected_count:
            sz_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✅ PASS: {fname} ({count:,} records, {sz_mb:.2f} MB)")
        else:
            print(f"  ❌ FAIL: {fname} has {count} records, expected {expected_count}")
            all_passed = False

    print("\n2. Checking Preprocessed Empty Document Fallback Recovery:")
    empty_ids = {'55497', '288457', '10533', '149317', '131890', '263763'}
    recovered = 0
    with open(processed_dir / 'corpus_preprocessed.jsonl', 'r', encoding='utf-8') as fh:
        for line in fh:
            doc = json.loads(line)
            if doc['doc_id'] in empty_ids:
                passage = doc.get('passage', '')
                passage_seg = doc.get('passage_segmented', '')
                if len(passage) > 0 and len(passage_seg) > 0:
                    recovered += 1
                    print(f"  ✅ Recovered empty doc `{doc['doc_id']}`: {passage[:60]}...")
                else:
                    print(f"  ❌ FAIL: Empty doc `{doc['doc_id']}` still has empty passage!")
                    all_passed = False
                    
    if recovered == len(empty_ids):
        print(f"  ✅ PASS: All {recovered} target empty docs successfully recovered with URL fallbacks!")
        
    print("\n3. Checking Train Target ID Coverage in Corpus:")
    corpus_ids = set()
    with open(processed_dir / 'corpus_preprocessed.jsonl', 'r', encoding='utf-8') as fh:
        for line in fh:
            doc = json.loads(line)
            corpus_ids.add(doc['doc_id'])
            
    missing_targets = 0
    with open(processed_dir / 'train_preprocessed.jsonl', 'r', encoding='utf-8') as fh:
        for line in fh:
            q = json.loads(line)
            for aid in q['answer_doc_ids']:
                if str(aid) not in corpus_ids:
                    missing_targets += 1
                    
    if missing_targets == 0:
        print("  ✅ PASS: 100% of training target document IDs exist in preprocessed corpus!")
    else:
        print(f"  ❌ FAIL: {missing_targets} target IDs missing from corpus!")
        all_passed = False

    print("\n4. Checking Documentation & Script Integrity:")
    docs_to_check = [
        'docs/MASTER_PLAN.md',
        'docs/data_audit_report.md',
        'docs/preprocessing_report.md',
        'docs/eda_report.md',
        'TASK_DISTRIBUTION.md',
        'requirements.txt',
        '.gitignore',
        '.gitattributes'
    ]
    for dpath in docs_to_check:
        fp = project_root / dpath
        if fp.exists():
            print(f"  ✅ PASS: {dpath} ({fp.stat().st_size} bytes)")
        else:
            print(f"  ❌ FAIL: {dpath} missing!")
            all_passed = False

    print("\n==================================================")
    if all_passed:
        print("    🎉 FULL AUDIT PASSED: ALL SYSTEMS GO FOR PHASE 4!  ")
    else:
        print("    ⚠️ AUDIT WARN: ISSUES DETECTED - SEE DETAILS ABOVE  ")
    print("==================================================")

if __name__ == '__main__':
    audit_all()
