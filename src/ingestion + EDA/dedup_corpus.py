"""
dedup_corpus.py — Corpus Deduplication via URL Page ID

Detects documents that are the same legal text crawled twice under different URL slugs.
thuvienphapluat.vn assigns a unique numeric page ID to each document, embedded in the URL:
    e.g. .../Luat-kham-benh-chua-benh-2023-so-15-2023-QH15-372143.aspx  <- page ID: 372143
         .../Luat-15-2023-QH15-kham-benh-chua-benh-372143.aspx          <- page ID: 372143

Both map to the same page — different slugs, same content.

OUTPUT: data/processed/dedup_mapping.json
    {
        "alias_doc_id": "canonical_doc_id",
        ...
    }

Canonical = doc with the LONGER passage (better quality crawl).
Alias docs are NOT deleted from corpus — only a mapping is produced for the retrieval layer.

Permitted under competition rules (muc 7): no corpus/label modifications,
only a system design decision about which doc to surface.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.paths import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("dedup_corpus")


def extract_page_id(link: str):
    if not link:
        return None
    m = re.search(r'-(\d+)\.aspx', link)
    return m.group(1) if m else None


def build_dedup_mapping(corpus_path: Path, output_path: Path) -> dict:
    """
    Scan corpus for duplicate documents sharing the same URL page ID.
    Returns a mapping of {alias_doc_id -> canonical_doc_id}.
    Canonical = doc with longer passage (better quality crawl).
    """
    page_id_groups = defaultdict(list)

    logger.info(f"Scanning {corpus_path} for duplicates...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            page_id = extract_page_id(r.get('link', ''))
            if page_id:
                page_id_groups[page_id].append({
                    'doc_id': r['doc_id'],
                    'passage_length': r.get('passage_length', 0),
                    'link': r.get('link', ''),
                })

    dup_groups = {pid: docs for pid, docs in page_id_groups.items() if len(docs) > 1}
    logger.info(f"Found {len(dup_groups)} duplicate groups covering {sum(len(v) for v in dup_groups.values())} documents")

    dedup_mapping = {}
    for page_id, docs in sorted(dup_groups.items()):
        docs_sorted = sorted(docs, key=lambda d: (-d['passage_length'], d['doc_id']))
        canonical = docs_sorted[0]
        aliases = docs_sorted[1:]
        for alias in aliases:
            dedup_mapping[alias['doc_id']] = canonical['doc_id']
        logger.info(
            f"  Page {page_id}: canonical={canonical['doc_id']} "
            f"(len={canonical['passage_length']:,}) | aliases={[a['doc_id'] for a in aliases]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dedup_mapping, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved dedup_mapping.json with {len(dedup_mapping)} alias entries")
    return dedup_mapping


def apply_dedup_to_predictions(predictions: list, dedup_mapping: dict) -> list:
    """
    Remap alias doc_ids in a prediction list to canonical, then deduplicate.
    Use this in the retrieval/evaluation layer after retrieving top-K results.
    """
    seen = set()
    remapped = []
    for doc_id in predictions:
        canonical = dedup_mapping.get(doc_id, doc_id)
        if canonical not in seen:
            seen.add(canonical)
            remapped.append(canonical)
    return remapped


def run_dedup():
    corpus_path = PROCESSED_DATA_DIR / 'corpus.jsonl'
    output_path = PROCESSED_DATA_DIR / 'dedup_mapping.json'

    if not corpus_path.exists():
        logger.error(f"corpus.jsonl not found at {corpus_path}. Run Phase 1 first.")
        return {}

    mapping = build_dedup_mapping(corpus_path, output_path)
    logger.info(f"=== DEDUP COMPLETE: {len(mapping)} alias doc_ids mapped to canonical ===")
    return mapping


if __name__ == '__main__':
    run_dedup()
