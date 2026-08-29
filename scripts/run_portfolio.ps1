# =============================================================================
# DANH MỤC CẤU HÌNH ĐỂ CHỌN BẰNG THỰC NGHIỆM TRÊN PUBLIC (chốt 2026-08-28)
#
# VÌ SAO LÀM THẾ NÀY: thước đo LOCAL đã chứng minh không xếp hạng được các ứng viên ở vùng
#   Δ < 0.01 — sàn nhiễu local là ±0.0132 (n=1398), và 3/3 lần Δ local dương dưới 0.01 đều ĐỔI DẤU
#   trên public (entity-boost, dòng #20, prior doc_type). Trong khi đó quy định mục 8 ghi rõ
#   "kết quả cuối lấy từ phương pháp cho điểm CAO NHẤT" -> thử thêm một cấu hình KHÁC BIỆT VỀ CƠ CHẾ
#   là việc KHÔNG CÓ MẶT TRÁI. Vậy nên: thôi đoán bằng local, đo thẳng trên phân phối thật.
#
# DANH MỤC (chỉ gồm các cấu hình khác nhau về CƠ CHẾ, không phải khác một chữ số hyperparameter):
#   #1 RRF baseline           v1=1 harrier=0.2 bm25=0.1 rrf_k=60      local 0.9319 | PUBLIC 0.9208 ✓
#   #2 CombSUM minmax         harrier=0.75 bm25=0.25                  local 0.9368 | chưa nộp
#   #3 CombSUM zscore         (chuẩn hoá khác, chưa đo cả local)      local  ?     | chưa nộp
#   #4 RRF + bge-m3 sparse    m3sparse=0.04                           local 0.9346 | chưa nộp
#   #5 CombSUM + m3sparse     (gộp 2 tín hiệu nhỏ, chưa đo local)     local  ?     | chưa nộp
#   (prior doc_type ĐÃ NỘP: 0.9193 < 0.9208 -> đã loại khỏi danh mục)
#
# THỨ TỰ CHẠY:
#   .\scripts\run_portfolio.ps1                      <- #2: sinh bài nộp NGAY (không GPU, ~1 phút)
#   .\scripts\run_portfolio.ps1 -Phase sparse-ranks  <- GPU: sinh ranks m3sparse CÓ SCORES (val+public)
#   .\scripts\run_portfolio.ps1 -Phase sparse-public <- #4: sinh bài nộp
#   .\scripts\run_portfolio.ps1 -Phase local         <- đo LOCAL cho #3 và #5 (chọn trọng số)
#   .\scripts\run_portfolio.ps1 -Phase public35 -ZHarrier .. -ZBm25 .. -SparseW ..   <- #3 và #5
# =============================================================================
param(
    [ValidateSet("combsum", "sparse-ranks", "sparse-public", "local", "public35")]
    [string]$Phase = "combsum",
    [double]$ZHarrier = 0.75,
    [double]$ZBm25 = 0.25,
    [double]$SparseW = 0.04
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$PubQ = "data/processed/public_test_preprocessed.jsonl"
$ValQ = "data/processed/val_preprocessed.jsonl"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}

# LUU Y: dat ten bien la $zipWork, KHONG phai $phase/$stage - trung ten tham so co ValidateSet
# se lam script chet ngay khi gan (da dinh dung loi nay 2026-08-28).
function New-SubmissionZip($jsonPath, $zipPath) {
    python -m src.evaluation.validate_submission --predictions $jsonPath
    Assert-Ok "validate $jsonPath"
    $zipWork = "outputs/predictions/_zip_work"
    if (Test-Path $zipWork) { Remove-Item $zipWork -Recurse -Force }
    New-Item -ItemType Directory -Path $zipWork | Out-Null
    Copy-Item $jsonPath (Join-Path $zipWork "submission.json")
    New-Item -ItemType Directory -Path "outputs/submissions" -Force | Out-Null
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path (Join-Path $zipWork "submission.json") -DestinationPath $zipPath
    Remove-Item $zipWork -Recurse -Force
    Write-Host "  -> $zipPath" -ForegroundColor Green
}

if ($Phase -eq "combsum") {
    Write-Host "`n=== #2 CombSUM minmax (harrier=0.75, bm25=0.25) tren PUBLIC ===" -ForegroundColor Cyan
    Write-Host "Khong can GPU: ranks_public_*.npz da co san CA scores (da kiem tra)." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion fuse --no-eval --mode score --score-norm minmax `
        --rank v1=outputs/cache/ranks_public_v1.npz `
        --rank harrier=outputs/cache/ranks_public_harrier.npz `
        --rank bm25=outputs/cache/ranks_public_bm25.npz `
        --weights harrier=0.75 --weights bm25=0.25 `
        --save-best-predictions outputs/predictions/submission_combsum.json
    Assert-Ok "fuse combsum public"
    New-SubmissionZip "outputs/predictions/submission_combsum.json" "outputs/submissions/submission_v10_combsum.zip"
    Write-Host "`nNOP FILE NAY, so voi 0.9208 (baseline da xac nhan)." -ForegroundColor Yellow
    Write-Host "Buoc tiep: .\scripts\run_portfolio.ps1 -Phase sparse-ranks" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "sparse-ranks") {
    Write-Host "`n=== Sinh ranks bge-m3 SPARSE CO SCORES cho CA val va public ===" -ForegroundColor Cyan
    Write-Host "Da sua bgem3_sparse.py de luu them key 'scores' (tuong thich nguoc: script RRF cu" -ForegroundColor DarkGray
    Write-Host "van doc duoc file nay binh thuong). Corpus da encode san 18 shard -> chi encode cau hoi." -ForegroundColor DarkGray

    Write-Host "`n--- val (1398 cau) — ghi de file cu de bo sung scores ---" -ForegroundColor Cyan
    python -m src.evaluation.bgem3_sparse rank --questions $ValQ `
        --out outputs/cache/ranks_bgem3sparse.npz
    Assert-Ok "sparse rank val"

    Write-Host "`n--- public test (1000 cau) ---" -ForegroundColor Cyan
    python -m src.evaluation.bgem3_sparse rank --questions $PubQ `
        --out outputs/cache/ranks_public_bgem3sparse.npz
    Assert-Ok "sparse rank public"

    Write-Host "`nBuoc tiep: .\scripts\run_portfolio.ps1 -Phase sparse-public" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "sparse-public") {
    Write-Host "`n=== #4 RRF + bge-m3 sparse (m3sparse=$SparseW) tren PUBLIC ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse --no-eval `
        --rank v1=outputs/cache/ranks_public_v1.npz `
        --rank harrier=outputs/cache/ranks_public_harrier.npz `
        --rank bm25=outputs/cache/ranks_public_bm25.npz `
        --rank m3sparse=outputs/cache/ranks_public_bgem3sparse.npz `
        --weights harrier=0.2 --weights bm25=0.1 --weights m3sparse=$SparseW --rrf-k 60 `
        --save-best-predictions outputs/predictions/submission_m3sparse.json
    Assert-Ok "fuse rrf+m3sparse public"
    New-SubmissionZip "outputs/predictions/submission_m3sparse.json" "outputs/submissions/submission_v11_rrf_m3sparse.zip"
    Write-Host "`nBuoc tiep: .\scripts\run_portfolio.ps1 -Phase local" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "local") {
    Write-Host "`n=== #3 CombSUM ZSCORE — do LOCAL, sweep quanh vung toi uu cua minmax ===" -ForegroundColor Cyan
    Write-Host "zscore it nhay voi 1-2 diem ngoai lai hon minmax; cua so trong so co the KHAC minmax" -ForegroundColor DarkGray
    Write-Host "nen quet rong hon, KHONG suy dien tu 0.75/0.25 cua minmax." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion fuse --mode score --score-norm zscore `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25.npz `
        --weights harrier=0,0.25,0.5,0.75,1.0,1.5 `
        --weights bm25=0,0.1,0.25,0.5 `
        --report outputs/predictions/combsum_zscore_report.json
    Assert-Ok "local zscore"

    Write-Host "`n=== #5 CombSUM minmax CO THEM m3sparse — do LOCAL, sweep trong so m3sparse ===" -ForegroundColor Cyan
    Write-Host "Neo harrier=0.75/bm25=0.25 (dinh minmax da biet), chi quet rieng m3sparse." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion fuse --mode score --score-norm minmax `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25.npz `
        --rank m3sparse=outputs/cache/ranks_bgem3sparse.npz `
        --weights harrier=0.75 --weights bm25=0.25 `
        --weights m3sparse=0,0.05,0.1,0.2,0.35,0.5,0.75 `
        --report outputs/predictions/combsum_m3sparse_report.json
    Assert-Ok "local combsum+m3sparse"

    Write-Host "`n=== CACH DOC ===" -ForegroundColor Yellow
    Write-Host "Moc so sanh: CombSUM minmax = 0.9368, RRF baseline = 0.9319." -ForegroundColor Yellow
    Write-Host "KHONG dung local de QUYET DINH nop hay khong (da chung minh la khong tin duoc o muc" -ForegroundColor Red
    Write-Host "Delta nay). Chi dung local de CHON TRONG SO trong moi cau hinh, roi nop het len public." -ForegroundColor Red
    Write-Host "Dong m3sparse=0 phai ra DUNG 0.9368 (sanity check)." -ForegroundColor Green
    Write-Host "`nBuoc tiep, thay so theo ket qua o tren:" -ForegroundColor Green
    Write-Host "  .\scripts\run_portfolio.ps1 -Phase public35 -ZHarrier <tot nhat> -ZBm25 <tot nhat> -SparseW <tot nhat>" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "public35") {
    Write-Host "`n=== #3 CombSUM zscore (harrier=$ZHarrier, bm25=$ZBm25) tren PUBLIC ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse --no-eval --mode score --score-norm zscore `
        --rank v1=outputs/cache/ranks_public_v1.npz `
        --rank harrier=outputs/cache/ranks_public_harrier.npz `
        --rank bm25=outputs/cache/ranks_public_bm25.npz `
        --weights harrier=$ZHarrier --weights bm25=$ZBm25 `
        --save-best-predictions outputs/predictions/submission_combsum_zscore.json
    Assert-Ok "fuse zscore public"
    New-SubmissionZip "outputs/predictions/submission_combsum_zscore.json" "outputs/submissions/submission_v12_combsum_zscore.zip"

    Write-Host "`n=== #5 CombSUM minmax + m3sparse=$SparseW tren PUBLIC ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse --no-eval --mode score --score-norm minmax `
        --rank v1=outputs/cache/ranks_public_v1.npz `
        --rank harrier=outputs/cache/ranks_public_harrier.npz `
        --rank bm25=outputs/cache/ranks_public_bm25.npz `
        --rank m3sparse=outputs/cache/ranks_public_bgem3sparse.npz `
        --weights harrier=0.75 --weights bm25=0.25 --weights m3sparse=$SparseW `
        --save-best-predictions outputs/predictions/submission_combsum_m3sparse.json
    Assert-Ok "fuse combsum+m3sparse public"
    New-SubmissionZip "outputs/predictions/submission_combsum_m3sparse.json" "outputs/submissions/submission_v13_combsum_m3sparse.zip"

    Write-Host "`n=== XONG DANH MUC ===" -ForegroundColor Green
    Write-Host "Nop het, ghi lai diem tung ban vao docs/submission_log_task1.md. Moc: 0.9208." -ForegroundColor Yellow
    exit 0
}
