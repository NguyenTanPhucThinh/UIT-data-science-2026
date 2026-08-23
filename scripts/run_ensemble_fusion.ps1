# =============================================================================
# ENSEMBLE nhieu model embedding + BM25 bang RRF nhieu nhanh (2026-08-21)
#
# Dung lai TOAN BO cache embedding co san -> KHONG encode lai corpus (tiet kiem hang chuc gio GPU).
# Pha 1 chay tuan tu tung model (KHONG song song - se no RAM). Pha 2 khong can GPU.
#
# Chay tu thu muc GOC:  .\scripts\run_ensemble_fusion.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== PHA 1a: thu hang cua v1 (AITeamVN, model chinh thuc) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name v1 `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --out outputs/cache/ranks_v1.npz

Write-Host "`n=== PHA 1b: thu hang cua vietlegal-harrier (bo tro tot nhat: cuu 34.1%) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name harrier `
    --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code `
    --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" `
    --max-seq-length 1024 `
    --out outputs/cache/ranks_harrier.npz

Write-Host "`n=== PHA 1c: thu hang cua bge-m3 goc (cuu 26.8%) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name bgem3 `
    --embedding-model BAAI/bge-m3 `
    --embedding-cache outputs/cache/embcompare_bge_m3.npz `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --out outputs/cache/ranks_bgem3.npz

Write-Host "`n=== PHA 1d: thu hang cua BM25 (khong can GPU/model) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 `
    --out outputs/cache/ranks_bm25.npz

Write-Host "`n=== PHA 2: fuse + sweep (KHONG dung GPU, chay lai bao nhieu lan cung duoc) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank bgem3=outputs/cache/ranks_bgem3.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --weights harrier=0,0.6,0.8,1.0,1.2 `
    --weights bgem3=0,0.6,0.8,1.0 `
    --weights bm25=0.1 `
    --rrf-k 60,30,10 `
    --report outputs/predictions/ensemble_sweep_report.json `
    --save-best-predictions outputs/predictions/ensemble_best.json

Write-Host ""
Write-Host "=== KIEM TRA BAT BUOC truoc khi tin bang ket qua ===" -ForegroundColor Green
Write-Host "Dong duoc danh dau '<-- baseline' (harrier=0 bgem3=0 bm25=0.1 rrf_k=60) PHAI ra Recall 0.9295." -ForegroundColor Green
Write-Host "Neu KHONG ra dung 0.9295 -> co gi do sai, BAO LAI cho Claude, dung tin bang do." -ForegroundColor Yellow
Write-Host ""
Write-Host "Neu to hop tot nhat vuot baseline >= +0.005 -> dang xac nhan tren public test." -ForegroundColor Green
Write-Host "NHO: phai cap nhat scripts/allowed-models.md + gui de xuat model moi cho BTC" -ForegroundColor Yellow
Write-Host "     (truoc 10 ngay so voi han chot private test) neu chot dung ensemble." -ForegroundColor Yellow
