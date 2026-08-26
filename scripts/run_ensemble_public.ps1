# =============================================================================
# TAO SUBMISSION PUBLIC bang ENSEMBLE v1 + harrier + BM25 (cap nhat 2026-08-25)
#
# Cau hinh DA CHOT tu sweep tren val_split_20 (KHONG sweep lai o day - public khong co dap an):
#     harrier = 0.2, rrf_k = 60, bm25 = 0.1, v1 = 1.0, dense/bm25-top-n = 100, gop = max
#     Score PUBLIC da xac nhan = 0.9208 (submission log dong #21).
#
# Chay tu thu muc GOC:  .\scripts\run_ensemble_public.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$Q = "data/processed/public_test_preprocessed.jsonl"

Write-Host "`n=== PHA 1a: thu hang v1 tren PUBLIC TEST ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name v1 `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --questions $Q `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --out outputs/cache/ranks_public_v1.npz

Write-Host "`n=== PHA 1b: thu hang harrier tren PUBLIC TEST ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name harrier `
    --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code `
    --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
    --questions $Q `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" `
    --max-seq-length 1024 `
    --out outputs/cache/ranks_public_harrier.npz

Write-Host "`n=== PHA 1c: thu hang BM25 tren PUBLIC TEST ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 `
    --questions $Q `
    --out outputs/cache/ranks_public_bm25.npz

Write-Host "`n=== PHA 2: fuse voi cau hinh DA CHOT (KHONG sweep) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion fuse --no-eval `
    --rank v1=outputs/cache/ranks_public_v1.npz `
    --rank harrier=outputs/cache/ranks_public_harrier.npz `
    --rank bm25=outputs/cache/ranks_public_bm25.npz `
    --weights harrier=0.2 `
    --weights bm25=0.1 `
    --rrf-k 60 `
    --save-best-predictions outputs/predictions/submission_ensemble.json

Write-Host "`n=== KIEM TRA DINH DANG truoc khi nop ===" -ForegroundColor Cyan
python -m src.evaluation.validate_submission --predictions outputs/predictions/submission_ensemble.json

Write-Host "`n=== DONG GOI submission.zip ===" -ForegroundColor Cyan
# File nop dat trong outputs/submissions/ - DUNG quy uoc san co cua du an (moi ban nop truoc deu o day),
# KHONG de trong outputs/predictions/ (thu muc do danh cho ket qua trung gian/eval).
$stage = "outputs/predictions/_zip_ensemble"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item outputs/predictions/submission_ensemble.json "$stage/submission.json"
New-Item -ItemType Directory -Path "outputs/submissions" -Force | Out-Null
$zip = "outputs/submissions/submission_v7_ensemble_harrier.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$stage/submission.json" -DestinationPath $zip
Remove-Item $stage -Recurse -Force

Write-Host ""
Write-Host "XONG. File nop: $zip" -ForegroundColor Green
Write-Host "  (ben trong chi chua duy nhat submission.json - dung quy dinh muc 2)" -ForegroundColor Green
Write-Host ""
Write-Host "Cau hinh da xac nhan PUBLIC 0.9208 (dong #21)." -ForegroundColor Yellow
