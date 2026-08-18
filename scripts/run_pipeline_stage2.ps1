# Giai đoạn 2 + 3 (train FULL trên toàn bộ train.json, rồi dự đoán + đóng gói submission.zip).
# CHỈ chạy sau khi Giai đoạn 1 (run_pipeline_stage1.ps1) đã cho thấy hybrid pipeline (embedding
# fine-tune + reranker + BM25) cải thiện rõ so với baseline 0.7806/0.1657 trên val_split_20.
#
# *** CAP NHAT (2026-08-13) - file nay chuyen vao thu muc con scripts\ (de gon thu muc goc) ***
# PHAI chay tu thu muc GOC UIT-data-science-2026 (khong phai tu ben trong scripts\) - vi tat ca
# duong dan trong script deu la duong dan tuong doi, tinh theo thu muc dang dung.
#
# Cách chạy: mở PowerShell tại thư mục UIT-data-science-2026 (thư mục GỐC), gõ:
#   .\scripts\run_pipeline_stage2.ps1
# Nếu bị chặn bởi execution policy, chạy:  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline_stage2.ps1
#
# *** CAP NHAT (2026-08-13) - da don lai thu muc outputs/ *** cache .npz -> outputs/cache/,
# prediction .json -> outputs/predictions/, file .zip da nop -> outputs/submissions/.

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

# Train FULL tren toan bo train.json (khong phai chi 80%). QUAN TRONG (2026-08-12, ban can nop
# GAP): dung DUNG cau hinh reranker da kiem chung o Giai doan 1 (--hard-negative-top-k 1 --epochs 1,
# cho ra Recall 0.8624/Precision 0.1838 tren val_split_20) - KHONG dung cau hinh nang hon
# (top-k 3/epochs 2) vi chua kiem chung va co the ton THEM VAI GIO nua, khong hop voi "can gap".
# Sau khi co ban nop dau tien, neu con thoi gian moi nen thu cau hinh nang hon de cai thien.
Invoke-Step "1/4 - Fine-tune embedding (FULL data)" `
    "python -m src.training.finetune_embedding --base-model bqbbao6/VN_legal_embedding_512 --train-json data/raw/train.json --corpus data/raw/selected-contexts --output-dir models/vietnamese-legal-embedding-ft-full" `
    "models/vietnamese-legal-embedding-ft-full"

Invoke-Step "2/4 - Fine-tune reranker (FULL data, hard negatives, cau hinh da kiem chung)" `
    "python -m src.training.finetune_reranker --base-model BAAI/bge-reranker-v2-m3 --train-json data/raw/train.json --corpus data/raw/selected-contexts --output-dir models/reranker-ft-full --hard-negatives --hard-negative-embedding-model models/vietnamese-legal-embedding-ft-full --hard-negative-top-k 1 --epochs 1" `
    "models/reranker-ft-full"

# Du doan tren public-official.json (BTC cham public leaderboard) - day la file se nop.
# QUAN TRONG (sua 2026-08-13, sau khi phat hien qua ket qua public thap hon han val 0.7571 vs
# 0.8624): PHAI dung public_test_preprocessed.jsonl (co question_segmented, khop cach tach tu
# voi corpus) - KHONG dung data/raw/public-official.json (tho, chua tach tu) - neu khong nhanh
# BM25 se khop tu kem chinh xac hon han so voi luc do tren val_preprocessed.jsonl, lam thap diem
# gia gia tao (khong phai do model kem, ma do lech tokenize giua cau hoi va corpus).
Invoke-Step "3/4 - Du doan tren public-official.json (dung ban da tach tu)" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model models/vietnamese-legal-embedding-ft-full --reranker-model models/reranker-ft-full --chunks data/processed/chunks_corpus.jsonl --questions data/processed/public_test_preprocessed.jsonl --output outputs/predictions/submission.json --embedding-cache outputs/cache/chunks_corpus_embeddings__ft_full.npz --batch-size 128"

# Kiem tra dinh dang truoc khi dong goi - BAT BUOC, tranh nop file sai dinh dang mat luot submit.
Invoke-Step "4/4 - Kiem tra dinh dang submission.json" `
    "python -m src.evaluation.validate_submission --predictions outputs/predictions/submission.json --corpus data/raw/selected-contexts --questions data/raw/public-official.json"

Write-Host ""
Write-Host "=== HOAN TAT - dong goi thu cong: nen CHI outputs/predictions/submission.json (khong kem thu muc con) thanh submission.zip trong outputs/submissions/ roi nop ===" -ForegroundColor Green
Write-Host '    Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission.zip -Force' -ForegroundColor DarkGray
