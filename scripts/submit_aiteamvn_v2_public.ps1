# Tao submission.zip cho AITeamVN/Vietnamese_Embedding_v2, bm25-weight=0.1, --bm25-fast.
#
# CHI CHAY SCRIPT NAY SAU KHI DA CHAY test_aiteamvn_v2_embedding.ps1 VA XEM SO LIEU LOCAL - dung
# ky luat cua du an (luon do LOCAL truoc khi ton luot nop PUBLIC). Neu Recall LOCAL cua v2 KHONG
# vuot duoc dong #11b (0.9303, v1 pretrained bm25=0.1) thi CAN NHAC KY co dang nop hay khong - xem
# lai bai hoc dong #17 (entity-boost): tin hieu LOCAL nho/am van co the (hiem) dao chieu tren PUBLIC,
# nhung nguoc lai tin hieu LOCAL am RO RANG thi hau nhu chac chan se khong thang tren PUBLIC.
#
# TAI SU DUNG cache outputs/cache/embcompare_aiteamvn_v2.npz - PHAI DA duoc tao boi
# test_aiteamvn_v2_embedding.ps1 truoc do (script nay se BAO LOI neu chua co, khong tu encode lai
# de tranh vo tinh cho ~1.5-2 tieng encode lai 351k chunk ma khong biet).
#
# Cach chay: mo PowerShell tai thu muc GOC UIT-data-science-2026, go:
#   .\scripts\submit_aiteamvn_v2_public.ps1

$ErrorActionPreference = "Stop"

$cachePath = "outputs\cache\embcompare_aiteamvn_v2.npz"
if (-not (Test-Path $cachePath)) {
    Write-Host "LOI: chua thay $cachePath - hay chay scripts\test_aiteamvn_v2_embedding.ps1 TRUOC (de tao cache tu val_preprocessed.jsonl, tranh encode lai 351k chunk 2 lan)." -ForegroundColor Red
    exit 1
}

Write-Host "=== 1/3: Tao hybrid predictions public test (AITeamVN/Vietnamese_Embedding_v2, bm25=0.1, --bm25-fast) ===" -ForegroundColor Cyan
python -m src.evaluation.full_pipeline_eval `
    --embedding-model AITeamVN/Vietnamese_Embedding_v2 `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/public_test_preprocessed.jsonl `
    --output outputs/predictions/submission_aiteamvn_v2.json `
    --embedding-cache outputs/cache/embcompare_aiteamvn_v2.npz `
    --stage-eval `
    --bm25-weight 0.1 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --bm25-fast

if ($LASTEXITCODE -ne 0) {
    Write-Host "LOI khi chay full_pipeline_eval - dung lai, khong tao submission." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 2/3: Dong goi submission.zip ===" -ForegroundColor Cyan
Copy-Item outputs\predictions\submission_aiteamvn_v2.hybrid_fused.json outputs\predictions\submission.json -Force
Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_aiteamvn_v2.zip -Force

Write-Host "`n=== XONG ===" -ForegroundColor Green
Write-Host "Da tao: outputs\submissions\submission_aiteamvn_v2.zip" -ForegroundColor Green
Write-Host "Nop file nay len he thong BTC, roi ghi lai Score PUBLIC vao submission-log-task1.md (dong moi," -ForegroundColor Yellow
Write-Host "so sanh voi 0.9173 = dong #11b, cau hinh dang dung hien tai voi v1)." -ForegroundColor Yellow
