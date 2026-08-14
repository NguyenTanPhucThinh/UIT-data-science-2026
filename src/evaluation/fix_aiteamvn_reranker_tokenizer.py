"""
Fix một lần cho lỗi "ValueError: Unrecognized processing class..." khi load
AITeamVN/Vietnamese_Reranker qua CrossEncoder (sentence-transformers) - bản thư viện mới dùng
AutoProcessor để load tokenizer, không tự hạ cấp xuống tokenizer "slow" (sentencepiece-only, không
có tokenizer.json) của model này được, dù bản thân AutoTokenizer một mình vẫn load được bình
thường (đã verify: tokenizer_class="XLMRobertaTokenizer" hợp lệ trong tokenizer_config.json).

Script này: tải model 1 lần, tự build tokenizer "fast" (có tokenizer.json - giống định dạng
BAAI/bge-reranker-v2-m3 đang chạy tốt), lưu CẢ model lẫn tokenizer ra 1 thư mục local. Từ đó dùng
đường dẫn local này làm --reranker-model thay vì "AITeamVN/Vietnamese_Reranker" trực tiếp.

Cách chạy (1 lần duy nhất, cần mạng để tải model ~2.3GB):
    python fix_aiteamvn_reranker_tokenizer.py

Sau đó dùng trong các lệnh full_pipeline_eval.py:
    --reranker-model models/AITeamVN_Vietnamese_Reranker_local
"""
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "AITeamVN/Vietnamese_Reranker"
OUT_DIR = Path("models/AITeamVN_Vietnamese_Reranker_local")

if __name__ == "__main__":
    if (OUT_DIR / "tokenizer.json").exists() and (OUT_DIR / "model.safetensors").exists():
        print(f"Đã có sẵn tại {OUT_DIR} (có tokenizer.json + model.safetensors) - bỏ qua, không tải lại.")
        raise SystemExit(0)

    print(f"Đang tải + convert tokenizer '{REPO}' sang bản fast (use_fast=True)...")
    tokenizer = AutoTokenizer.from_pretrained(REPO, use_fast=True)
    print("  -> OK, class:", type(tokenizer).__name__)

    print(f"Đang tải model weights '{REPO}' (~2.3GB, có thể mất vài phút)...")
    model = AutoModelForSequenceClassification.from_pretrained(REPO)
    print("  -> OK, class:", type(model).__name__)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Đang lưu cả 2 vào {OUT_DIR} ...")
    tokenizer.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)

    if not (OUT_DIR / "tokenizer.json").exists():
        print("CẢNH BÁO: không thấy tokenizer.json sau khi save - convert fast tokenizer có thể đã thất bại.")
        raise SystemExit(1)

    print(f"\nHOÀN TẤT. Từ giờ dùng --reranker-model {OUT_DIR} thay vì '{REPO}' trực tiếp.")
