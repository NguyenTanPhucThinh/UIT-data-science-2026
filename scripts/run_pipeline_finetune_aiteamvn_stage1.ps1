# Giai đoạn 1 - Fine-tune AITeamVN/Vietnamese_Embedding tren 80% train, danh gia sach tren val_split_20
# (khong ro ri du lieu). KHONG dung reranker (da chot bo reranker khoi cau hinh dang dung - xem
# submission-log-task1.md muc "QUYET DINH HIEN TAI").
#
# *** CAP NHAT (2026-08-15) - them --fp16 cho buoc fine-tune (mixed-precision) ***
# Neu ban dang chay ban CU cua script nay (khong co --fp16) va thay toc do rat cham (vd 70+ giay/it,
# ETA hang chuc tieng cho 1146 step) - DUNG lai (Ctrl+C), roi chay lai file .ps1 MOI nay. Nguyen nhan
# nhieu kha nang la sentence-transformers MAC DINH train fp32 (khong bat mixed-precision) - cong voi
# AITeamVN/Vietnamese_Embedding la model LON hon han (~568M, kien truc XLM-R-large 24 lop) so voi
# bqbbao6 truoc day (~278M, base 12 lop) VA --max-seq-length dai hon nhieu (2048 so voi ~512) - chi
# phi attention tang theo BINH PHUONG do dai chuoi. --fp16 (bat use_amp=True trong model.fit()) giup
# nhanh hon ~1.5-2x tren GPU NVIDIA doi moi (Turing/Ampere tro len), chat luong gan nhu khong doi.
# LUU Y: so lieu "it/s" hien o BUOC DAU TIEN thuong bi thoi phong rat nhieu (CUDA/cudnn autotune +
# bien dich kernel lan dau) - da tung gap hien tuong nay khi encode corpus (progress bar ban dau
# uoc luong ~7 tieng nhung sau do nhanh han han) - neu van muon kiem tra truoc khi doi --fp16, de
# script chay toi step ~10-20 roi xem lai it/s trung binh (khong chi step 1) truoc khi ket luan.
#
# Neu bi chan boi execution policy ("scripts is disabled on this system"), chay:
#   powershell -ExecutionPolicy Bypass -File run_pipeline_finetune_aiteamvn_stage1.ps1
# Chay tu thu muc GOC UIT-data-science-2026 (duong dan trong script la duong dan tuong doi).

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

# Buoc 1/3 - Fine-tune embedding (80% train, in-batch negatives, KHONG hard-negatives vi khong
# dung reranker). --query-prefix=""/--passage-prefix="" (dang dinh lien, tranh bug PowerShell nuot
# tham so chuoi rong). --max-seq-length 2048 khop voi khuyen nghi model card VA khop voi gia tri
# se dung luc inference (full_pipeline_eval.py buoc 2 ben duoi) - PHAI giong nhau. --fp16 de tang
# toc mixed-precision training.
Invoke-Step "1/3 - Fine-tune AITeamVN/Vietnamese_Embedding (80%, in-batch negatives, fp16)" `
    "python -m src.training.finetune_embedding --base-model AITeamVN/Vietnamese_Embedding --train-json data/processed/train_split_80.json --corpus data/raw/selected-contexts --output-dir models/vietnamese-embedding-aiteamvn-ft --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048 --fp16" `
    "models/vietnamese-embedding-aiteamvn-ft"

# Buoc 2/3 - Danh gia tren val_split_20 (khong ro ri, KHONG dung reranker, bm25-weight=0.4 - giong
# het cau hinh dang dung voi ban pretrained, chi doi embedding-model sang ban vua fine-tune).
Invoke-Step "2/3 - Pipeline hybrid tren tap validate (20%), KHONG rerank, bm25-weight=0.4" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model models/vietnamese-embedding-aiteamvn-ft --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/aiteamvn_ft_stage1.json --embedding-cache outputs/cache/aiteamvn_ft_stage1.npz --bm25-weight 0.4 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048"

# Buoc 3/3 - Cham diem, luu report rieng de so sanh voi ban pretrained (dong #11).
Invoke-Step "3/3 - Cham diem (Recall/Precision chinh thuc BTC)" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/aiteamvn_ft_stage1.json --report outputs/predictions/aiteamvn_ft_stage1_report.json"

Write-Host ""
Write-Host "=== HOAN TAT GIAI DOAN 1 (fine-tune AITeamVN) - so Recall/Precision o tren voi moc PRETRAINED 0.9267/0.1973 (dong #11) ===" -ForegroundColor Green
Write-Host "Neu Recall tang ro ret (vd +0.02 tro len) so voi 0.9267 - dang chuan bi Giai doan 2 (fine-tune FULL 100% train.json)." -ForegroundColor Green
Write-Host "Neu Recall KHONG tang hoac giam - co the day la truong hop model da du manh o dang pretrained, fine-tune them khong giup nhieu (khac voi bqbbao6 truoc day) - bao lai ket qua de quyet dinh buoc tiep theo." -ForegroundColor Yellow
