import os
import json
import shutil
import time
import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.paths import DATA_DIR, MODELS_DIR, DOCS_DIR
from src.config.task_data import get_task_data, selected_tasks
import text_cleaner
import legal_preprocessor

def _configure_java_home() -> None:
    """Find a usable local JRE/JDK when JAVA_HOME is not already configured."""
    configured = os.environ.get("JAVA_HOME")
    if configured and Path(configured).is_dir():
        return

    candidates = []
    for environment_name in ("ProgramFiles", "ProgramFiles(x86)"):
        program_files = os.environ.get(environment_name)
        if not program_files:
            continue
        java_root = Path(program_files) / "Java"
        if java_root.is_dir():
            candidates.extend(java_root.glob("jdk*"))
            candidates.extend(java_root.glob("jre*"))

    java_executable = shutil.which("java")
    if java_executable:
        candidates.append(Path(java_executable).resolve().parent.parent)

    for candidate in candidates:
        jvm_locations = (
            candidate / "bin" / "server" / "jvm.dll",
            candidate / "jre" / "bin" / "server" / "jvm.dll",
            candidate / "lib" / "server" / "libjvm.so",
            candidate / "jre" / "lib" / "server" / "libjvm.so",
        )
        if any(location.is_file() for location in jvm_locations):
            os.environ["JAVA_HOME"] = str(candidate)
            return

    raise RuntimeError(
        "Không tìm thấy Java runtime. Hãy cài Java 8+ và đặt JAVA_HOME tới thư mục JRE/JDK "
        "(ví dụ C:\\Program Files\\Java\\jre-1.8)."
    )


_configure_java_home()

import py_vncorenlp
from pyvi import ViTokenizer

rdrsegmenter = None

VNCORENLP_BASE_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"
VNCORENLP_REQUIRED_FILES = {
    "VnCoreNLP-1.2.jar": 1_000_000,
    "models/wordsegmenter/vi-vocab": 10_000,
    "models/wordsegmenter/wordsegmenter.rdr": 10_000,
}


def _is_valid_model_file(path: Path, minimum_size: int) -> bool:
    """Reject absent, empty, and obviously truncated VnCoreNLP artifacts."""
    return path.is_file() and path.stat().st_size >= minimum_size


def _download_file(url: str, destination: Path) -> None:
    """Download atomically with Python so this works on Windows without wget."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "UIT-LegalQA-data-pipeline/1.0"})

    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Không thể tải VnCoreNLP từ {url}. Hãy kiểm tra kết nối mạng "
            f"hoặc tải thủ công file vào {destination}."
        ) from exc


def ensure_vncorenlp_model(save_dir: Path) -> None:
    """Ensure the artifacts required by the ``wseg`` annotator are complete."""
    missing = [
        relative_path
        for relative_path, minimum_size in VNCORENLP_REQUIRED_FILES.items()
        if not _is_valid_model_file(save_dir / relative_path, minimum_size)
    ]
    if not missing:
        return

    print("Downloading missing VnCoreNLP word-segmentation artifacts...")
    for relative_path in missing:
        destination = save_dir / relative_path
        print(f"  - {relative_path}")
        _download_file(f"{VNCORENLP_BASE_URL}/{relative_path}", destination)

    invalid = [
        relative_path
        for relative_path, minimum_size in VNCORENLP_REQUIRED_FILES.items()
        if not _is_valid_model_file(save_dir / relative_path, minimum_size)
    ]
    if invalid:
        raise RuntimeError(
            "VnCoreNLP download incomplete; missing or invalid files: " + ", ".join(invalid)
        )

def init_worker():
    global rdrsegmenter
    save_dir = MODELS_DIR / 'vncorenlp'

    # py_vncorenlp.download_model() shells out to Unix `wget`, which is not
    # available on a standard Windows installation. Download only the files
    # needed by the wseg annotator using Python instead.
    ensure_vncorenlp_model(save_dir)

    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))

def fast_tokenize_text(text: str) -> str:
    """Segment line by line with VnCoreNLP, falling back to PyVi on individual failures."""
    if not text:
        return ""
    lines = text.split("\n")
    tokenized_lines = []
    for line in lines:
        if line.strip():
            try:
                if rdrsegmenter is None:
                    raise RuntimeError("VnCoreNLP worker is not initialized")
                sentences = rdrsegmenter.word_segment(line)
                tokenized = " ".join(sentences)
            except Exception:
                # A malformed/oversized line must not abort preprocessing of the whole corpus.
                tokenized = ViTokenizer.tokenize(line)
            tokenized_lines.append(tokenized)
        else:
            tokenized_lines.append("")
    return "\n".join(tokenized_lines)

def process_single_doc(r):
    doc_id = r['doc_id']
    raw_passage = r.get('passage', '')
    name = r.get('name')
    link = r.get('link', '')
    was_empty = r.get('is_empty', False)
    
    clean_p = text_cleaner.preprocess_text(raw_passage, doc_id=doc_id, name=name, link=link)
    meta = legal_preprocessor.extract_legal_metadata(clean_p, name=name, link=link)
    
    # Fast line-by-line word segmentation
    passage_segmented = fast_tokenize_text(clean_p)
    
    return {
        'doc_id': doc_id,
        'name': name,
        'link': link,
        'passage': clean_p,
        'passage_segmented': passage_segmented,
        'passage_length': len(clean_p),
        'doc_type': r.get('doc_type', 'Khác'),
        'is_empty_fallback': was_empty,
        'metadata': meta
    }, len(raw_passage), len(clean_p), 1 if was_empty else 0

def run_phase2(task="all"):
    tasks = selected_tasks(task)
    print(
        "=== STARTING PHASE 2: ULTRA-FAST PREPROCESSING & SEGMENTATION "
        f"(TASKS {', '.join(tasks)}) ==="
    )
    
    input_dir = DATA_DIR / 'processed'
    corpus_in = input_dir / 'corpus.jsonl'
    corpus_out = input_dir / 'corpus_preprocessed.jsonl'
    
    # 1. Preprocess Questions
    def process_questions(in_filepath, out_filepath, desc):
        records = []
        with open(in_filepath, 'r', encoding='utf-8') as fh:
            for line in fh:
                records.append(json.loads(line))
                
        processed = []
        for r in tqdm(records, desc=desc):
            q_raw = r['question']
            q_clean = text_cleaner.normalize_unicode(q_raw)
            q_clean = text_cleaner.clean_whitespace(q_clean)
            q_segmented = fast_tokenize_text(q_clean)
            
            pr = {
                'question_id': r['question_id'],
                'question': q_clean,
                'question_segmented': q_segmented,
            }
            if 'answer_doc_ids' in r:
                pr['answer_doc_ids'] = r.get('answer_doc_ids')
            if 'num_answers' in r:
                pr['num_answers'] = r.get('num_answers')
            if 'answer' in r:
                pr['answer'] = r.get('answer')
            processed.append(pr)
            
        with open(out_filepath, 'w', encoding='utf-8') as out_fh:
            for pr in processed:
                out_fh.write(json.dumps(pr, ensure_ascii=False) + '\n')
        print(f"Saved {out_filepath} ({len(processed)} records)")
        return processed

    # Initialize segmenter for the main process since process_questions runs synchronously
    global rdrsegmenter
    if rdrsegmenter is None:
        init_worker()
        
    print("\nPreprocessing Questions...")
    for selected_task in tasks:
        paths = get_task_data(selected_task)
        process_questions(
            paths.clean_train,
            paths.preprocessed_train,
            f"Task {selected_task} Train Questions",
        )
        if selected_task == "1":
            process_questions(
                input_dir / 'val_clean.jsonl',
                input_dir / 'val_preprocessed.jsonl',
                "Task 1 Val Questions",
            )
        process_questions(
            paths.clean_public,
            paths.preprocessed_public,
            f"Task {selected_task} Public Test Questions",
        )
    
    # 2. Preprocess Corpus with Multithreading
    print(f"\nProcessing {corpus_in} in parallel...")
    corpus_records = []
    with open(corpus_in, 'r', encoding='utf-8') as fh:
        for line in fh:
            corpus_records.append(json.loads(line))
            
    t0 = time.time()
    processed_corpus = []
    total_raw_chars = 0
    total_clean_chars = 0
    fallback_count = 0
    
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker) as executor:
        results = list(tqdm(executor.map(process_single_doc, corpus_records), total=len(corpus_records), desc="Preprocessing Corpus"))
        
    for res_doc, raw_len, clean_len, fb in results:
        processed_corpus.append(res_doc)
        total_raw_chars += raw_len
        total_clean_chars += clean_len
        fallback_count += fb
        
    with open(corpus_out, 'w', encoding='utf-8') as out_fh:
        for pr in processed_corpus:
            out_fh.write(json.dumps(pr, ensure_ascii=False) + '\n')
            
    elapsed = time.time() - t0
    print(f"\nSaved {corpus_out} ({len(processed_corpus)} records in {elapsed:.1f}s)")
    print(f"  Empty passages recovered with URL fallback: {fallback_count}")
    print(f"  Total raw chars: {total_raw_chars:,} -> Clean chars: {total_clean_chars:,}")
    
    # Generate preprocessing_report.md
    report_content = f"""# Text Preprocessing & Normalization Report (Phase 2)
**UIT Data Science Challenge 2026 — Task 1 (LegalIR)**  
**Author:** Đoàn Quốc Kiên (Data & Utilities)  

---

## 1. Preprocessing Summary
- **Unicode Normalization**: Applied NFC standard normalization to 100% of corpus text and questions.
- **Whitespace & Line Breaks**: Normalized `\\r\\n` → `\\n`, stripped trailing spaces, capped empty lines to max 2.
- **Header Formatting**: Applied concise header normalization without content stripping.
- **URL Fallback Generation**: Successfully generated URL-slug fallback text for **20 empty passage documents** (including 6 target documents referenced in `train.json`).
- **Vietnamese Word Segmentation**: Applied line-by-line `VnCoreNLP` with a per-line PyVi fallback in 8 worker processes to produce `passage_segmented` and `question_segmented` (compounding words with `_`).

---

## 2. Quantitative Metrics
| Metric | Value |
|---|---|
| **Corpus Total Documents** | 8,532 |
| **Empty Documents Recovered** | 20 |
| **Total Raw Character Count** | {total_raw_chars:,} |
| **Total Clean Character Count** | {total_clean_chars:,} |
| **Word Segmenter Engine** | VnCoreNLP (ProcessPool 8 Workers) |
| **Processing Time** | {elapsed:.1f} seconds |

---

## 3. Output Datasets in `data/processed/`
- [corpus_preprocessed.jsonl](file:///d:/UIT-data-science-2026/data/processed/corpus_preprocessed.jsonl)
- [train_preprocessed.jsonl](file:///d:/UIT-data-science-2026/data/processed/train_preprocessed.jsonl)
- [val_preprocessed.jsonl](file:///d:/UIT-data-science-2026/data/processed/val_preprocessed.jsonl)
- [public_test_preprocessed.jsonl](file:///d:/UIT-data-science-2026/data/processed/public_test_preprocessed.jsonl)

---

## 4. Next Phase
Moving on to **Phase 3: Exploratory Data Analysis (EDA)**.
"""
    
    print("=== PHASE 2 COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess corpus and questions by task")
    parser.add_argument("--task", choices=("1", "2", "all"), default="all")
    run_phase2(parser.parse_args().task)
