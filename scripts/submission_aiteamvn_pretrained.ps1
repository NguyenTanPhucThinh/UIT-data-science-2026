# Tao submission.zip cho AITeamVN/Vietnamese_Embedding PRETRAINED (CHUA fine-tune), cau hinh dong
# #11 trong claude/submission-log-task1.md (Recall LOCAL 0.9267, KHONG reranker, bm25-weight=0.4).
# Muc dich: xac nhan cai thien +0.0348 so voi dong #10 co chuyen thanh cai thien tuong ung tren
# public leaderboard hay khong - day la cai thien LON (khac han muc +0.01 nho cua bm25-weight truoc
# day), nen dang nop rieng de xac nhan (khong chi gia dinh).
#
# TAI SU DUNG cache da co san outputs/cache/embcompare_aiteamvn.npz (sinh ra khi chay
# run_pipeline_test_aiteamvn_embedding.ps1 tren val_preprocessed.jsonl) - cache chi phu thuoc
# model + chunks_corpus.jsonl, KHONG phu thuoc bo cau hoi, nen dung LAI duoc cho public_test - KHONG
# can encode lai 351k chunk lan nua (tiet kiem hang chuc phut).
#
# QUAN TRONG: script nay CHI sinh submission_run_N.json/zip - BAN PHAI TU TAY nop file .zip nay len
# he thong nop bai cua BTC (Claude khong co quyen truy cap he thong do), roi ghi lai Score PUBLIC +
# submission ID vao claude/submission-log-task1.md (bao lai cho Claude de cap nhat, hoac tu sua).
#
# Cach chay: mo PowerShell tai thu muc GOC UIT-data-science-2026, go:
#   .\submission_aiteamvn_pretrained.ps1
# Neu bi chan boi execution policy:
#   powershell -ExecutionPolicy Bypass -File submission_aiteamvn_pretrained.ps1

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

Invoke-Step "1/3 - Chay full pipeline tren public_test_preprocessed.jsonl (AITeamVN pretrained, KHONG reranker, bm25-weight=0.4)" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model AITeamVN/Vietnamese_Embedding --chunks data/processed/chunks_corpus.jsonl --questions data/processed/public_test_preprocessed.jsonl --output outputs/predictions/submission_v4_aiteamvn_pretrained.json --embedding-cache outputs/cache/embcompare_aiteamvn.npz --stage-eval --bm25-weight 0.4 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048" `
    "outputs/predictions/submission_v4_aiteamvn_pretrained.hybrid_fused.json"

Invoke-Step "2/3 - Sao chep file hybrid_fused.json thanh submission.json (dung file nay - KHONG dung file goc submission_v4_aiteamvn_pretrained.json vi khong co reranker nen 2 file giong nhau, nhung theo quy uoc du an luon lay .hybrid_fused.json cho ro rang)" `
    "Copy-Item outputs\predictions\submission_v4_aiteamvn_pretrained.hybrid_fused.json outputs\predictions\submission.json -Force"

Invoke-Step "3/3 - Nen thanh submission_v4_aiteamvn_pretrained.zip" `
    "Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_v4_aiteamvn_pretrained.zip -Force"

Write-Host ""
Write-Host "=== HOAN TAT - file san sang tai outputs\submissions\submission_v4_aiteamvn_pretrained.zip ===" -ForegroundColor Green
Write-Host "    BUOC TIEP THEO (BAN TU LAM): dang nhap he thong nop bai BTC, upload file .zip nay." -ForegroundColor Yellow
Write-Host "    Sau khi co Score PUBLIC + submission ID, bao lai cho Claude (hoac tu ghi) vao claude/submission-log-task1.md dong #11." -ForegroundColor DarkGray
