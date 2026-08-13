# Giai đoạn 1 (train & evaluate LOCAL trên 80/20) - chạy 1 lần, tự dừng nếu có bước lỗi.
# Cách chạy: mở PowerShell tại thư mục UIT-data-science-2026, gõ:  .\run_pipeline_stage1.ps1
# Nếu bị chặn bởi execution policy ("scripts is disabled on this system"), chạy:
#   powershell -ExecutionPolicy Bypass -File run_pipeline_stage1.ps1
#
# *** QUAN TRỌNG (2026-08-12) - đã sửa bug thiếu prefix "query: "/"passage: " *** bqbbao6/VN_legal_embedding_512
# fine-tune từ intfloat/multilingual-e5-base (bi-encoder E5-style, đã xác nhận qua model card
# HuggingFace) - BẮT BUỘC prefix này để encode đúng, nếu không Recall bị đánh giá thấp hơn thực
# tế. finetune_embedding.py/baseline_retrieval.py/full_pipeline_eval.py giờ TỰ ĐỘNG thêm prefix
# (mặc định, không cần sửa gì thêm). NHƯNG script này có logic tự BỎ QUA bước train nếu
# models/vietnamese-legal-embedding-ft đã có sẵn file - nếu bạn đã train model đó TRƯỚC khi có
# fix này (model cũ KHÔNG có prefix), phải XÓA thư mục models/vietnamese-legal-embedding-ft
# (và xóa luôn outputs/chunks_corpus_embeddings__ft.npz nếu có) rồi chạy lại, nếu không script sẽ
# tự bỏ qua và tiếp tục dùng model cũ/cache cũ (sai). Reranker (CrossEncoder) KHÔNG bị ảnh hưởng
# bởi prefix này - model reranker cũ vẫn dùng được bình thường, không cần train lại.

# Cac buoc co --output-dir: neu thu muc do da co san file (model da train xong tu lan chay
# truoc bi ngat giua chung), TU DONG BO QUA - khong train lai tu dau, tiet kiem thoi gian.
function Invoke-Step($desc, $cmd, $skipIfExists = $null) {
    Write-Host ""
    if ($skipIfExists -and (Test-Path $skipIfExists) -and (Get-ChildItem $skipIfExists -ErrorAction SilentlyContinue)) {
        Write-Host "=== $desc (BO QUA - da co san tai $skipIfExists) ===" -ForegroundColor Yellow
        return
    }
    Write-Host "=== $desc ===" -ForegroundColor Cyan
    Write-Host $cmd -ForegroundColor DarkGray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "LOI o buoc: $desc (exit code $LASTEXITCODE) - dung lai, khong chay tiep buoc sau." -ForegroundColor Red
        exit 1
    }
}

Invoke-Step "1/4 - Fine-tune embedding (80%)" `
    "python -m src.training.finetune_embedding --base-model bqbbao6/VN_legal_embedding_512 --train-json data/processed/train_split_80.json --corpus data/raw/selected-contexts --output-dir models/vietnamese-legal-embedding-ft" `
    "models/vietnamese-legal-embedding-ft"

# hard-negative-top-k 1 + epochs 1: rut gon cho vong thu nghiem dau (CrossEncoder nang hon
# embedding model nhieu - fp32 mac dinh + du lieu day du co the mat 70-80 tieng). Khi da chon
# duoc cau hinh tot o Giai doan 1, tang lai --hard-negative-top-k 3 --epochs 2 cho Giai doan 2
# (train full, chi chay dung 1 lan).
Invoke-Step "2/4 - Fine-tune reranker (80%, hard negatives, rut gon)" `
    "python -m src.training.finetune_reranker --base-model BAAI/bge-reranker-v2-m3 --train-json data/processed/train_split_80.json --corpus data/raw/selected-contexts --output-dir models/reranker-ft --hard-negatives --hard-negative-embedding-model models/vietnamese-legal-embedding-ft --hard-negative-top-k 1 --epochs 1" `
    "models/reranker-ft"

Invoke-Step "3/4 - Pipeline hybrid tren tap validate (20%)" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model models/vietnamese-legal-embedding-ft --reranker-model models/reranker-ft --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/full_pipeline_predictions.json --embedding-cache outputs/chunks_corpus_embeddings__ft.npz"

Invoke-Step "4/4 - Cham diem (Recall/Precision chinh thuc BTC)" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/full_pipeline_predictions.json"

Write-Host ""
Write-Host "=== HOAN TAT GIAI DOAN 1 - xem so Recall/Precision o tren, so voi moc dense-only 0.7806/0.1657 ===" -ForegroundColor Green
