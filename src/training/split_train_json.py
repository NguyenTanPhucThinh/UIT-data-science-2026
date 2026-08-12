"""
Áp dụng bộ chia train/val 80/20 chung của nhóm (train_val_split.json, do Kiên - Data & Utilities
tạo ở Phase 1) lên train.json, sinh ra 2 file CÙNG SCHEMA với train.json:

  - train_split_80.json  (~5.600 câu hỏi, dùng để FINE-TUNE)
  - val_split_20.json    (~1.400 câu hỏi, dùng để VALIDATE, KHÔNG được dùng để train)

Lý do làm vậy thay vì sửa từng script (finetune_embedding.py / baseline_retrieval.py /
evaluate.py): các script đó vốn đã nhận một file "hình dạng train.json" qua --train-json /
--questions, nên chỉ cần trỏ chúng sang 2 file output của script này là đủ - không cần thêm
tham số --split/--qid-file vào từng nơi.

Cách dùng:
    python -m src.training.split_train_json \\
        --train-json data/raw/train.json \\
        --split-file data/processed/train_val_split.json \\
        --output-dir data/processed

Sau đó:
    # 1) Fine-tune CHỈ trên 80%
    python -m src.training.finetune_embedding --train-json data/processed/train_split_80.json ...

    # 2) Validate model vừa fine-tune trên 20% còn lại (chưa từng thấy khi train)
    python -m src.evaluation.baseline_retrieval --questions data/processed/val_split_20.json ...
    python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions ...

    # 3) Khi đã chọn được cấu hình tốt nhất qua bước 1-2, train lại bản CUỐI CÙNG để nộp
    #    submission bằng data ĐẦY ĐỦ (không lọc gì cả) - dùng thẳng train.json gốc:
    python -m src.training.finetune_embedding --train-json data/raw/train.json ...
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def split_train_json(train_json_path: str, split_file_path: str) -> tuple[dict, dict]:
    with open(train_json_path, encoding="utf-8") as f:
        train_full: dict = json.load(f)

    with open(split_file_path, encoding="utf-8") as f:
        split: dict = json.load(f)

    train_qids = set(split["train_qids"])
    val_qids = set(split["val_qids"])

    missing = (train_qids | val_qids) - set(train_full.keys())
    if missing:
        raise ValueError(
            f"{len(missing)} qid trong split-file không tồn tại trong train.json "
            f"(vd: {list(missing)[:5]}) - kiểm tra lại 2 file có cùng phiên bản train.json không."
        )

    train_split = {qid: item for qid, item in train_full.items() if qid in train_qids}
    val_split = {qid: item for qid, item in train_full.items() if qid in val_qids}

    logger.info(
        "Chia xong: %d câu train (80%%), %d câu val (20%%), seed=%s, val_ratio=%s",
        len(train_split), len(val_split), split.get("seed"), split.get("val_ratio"),
    )
    return train_split, val_split


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-json", required=True, help="data/raw/train.json (bản đầy đủ, 7000 câu)")
    parser.add_argument("--split-file", required=True, help="data/processed/train_val_split.json (của Kiên)")
    parser.add_argument("--output-dir", required=True, help="Thư mục lưu train_split_80.json / val_split_20.json")
    args = parser.parse_args()

    train_split, val_split = split_train_json(args.train_json, args.split_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_split_80.json"
    val_out = output_dir / "val_split_20.json"

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_split, f, ensure_ascii=False, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_split, f, ensure_ascii=False, indent=2)

    logger.info("Đã lưu: %s (%d câu)", train_out, len(train_split))
    logger.info("Đã lưu: %s (%d câu)", val_out, len(val_split))


if __name__ == "__main__":
    main()
