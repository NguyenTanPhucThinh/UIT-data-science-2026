# =============================================================================
# NỘP THỬ PRIOR LOẠI VĂN BẢN (doc_type) — hướng mới sau chẩn đoán 2026-08-28.
#
# BỐI CẢNH: chẩn đoán (diagnose_ceiling.py) cho thấy 85.7% lỗi còn lại là lỗi XẾP HẠNG chứ không
#   phải lỗi truy hồi (tài liệu đúng đã nằm trong pool ở 99% số câu). Prior theo doc_type là tín
#   hiệu CẤP DOCUMENT, không phụ thuộc câu hỏi — phạm trù hoàn toàn khác 15 hướng ngữ nghĩa đã thử.
#   Trên val: beta=0.0005 -> Recall 0.9390 (+0.0072), cứu 10 câu / hỏng 1 câu, McNemar p=0.0117.
#   Đây là kết quả CÓ Ý NGHĨA THỐNG KÊ đầu tiên của dự án.
#
# ⚠️ VÌ SAO PHẢI SINH LẠI ranks_public VÀ NỘP CẢ 2 BẢN:
#   Đã xác minh (28-08): outputs/predictions/submission_ensemble.json (artifact 21-08, được cho là
#   bản đạt PUBLIC 0.9208) KHÔNG tái lập được từ code + cache hiện tại. Đã thử toàn bộ lưới
#   (top-n 50/100/150/200 x rrf_k=60 x 3 bộ trọng số) — khớp cao nhất chỉ 271/1000 câu, và 550/1000
#   câu khác NHAU CẢ TẬP tài liệu, không chỉ thứ tự. Kết luận: artifact đó sinh từ ranks CŨ HƠN
#   ranks_public_*.npz hiện có. => KHÔNG được dùng 0.9208 làm mốc so sánh cho bản nộp mới.
#   Cách xử lý đúng: sinh lại MỌI THỨ từ trạng thái hiện tại, rồi nộp CẢ baseline (beta=0) LẪN bản
#   prior, để Δ public đo được là Δ SẠCH giữa 2 bản cùng sinh từ một trạng thái.
#
# CÁCH DÙNG (chạy từ thư mục GỐC repo):
#   .\scripts\run_doctype_prior_submit.ps1                 <- đầy đủ: sinh lại ranks + 2 bài nộp
#   .\scripts\run_doctype_prior_submit.ps1 -Stage package  <- bỏ qua encode, dùng ranks_public có sẵn
# =============================================================================
param(
    [ValidateSet("full", "package")]
    [string]$Stage = "full",
    [double]$Beta = 0.0005
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$Q = "data/processed/public_test_preprocessed.jsonl"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}

if ($Stage -eq "full") {
    Write-Host "`n=== [1/4] SINH LAI thu hang PUBLIC TEST tu trang thai HIEN TAI ===" -ForegroundColor Cyan
    Write-Host "Chi encode 1000 cau hoi public test (dung cache corpus co san) - vai phut." -ForegroundColor DarkGray

    Write-Host "`n--- v1 ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank --name v1 `
        --embedding-model AITeamVN/Vietnamese_Embedding `
        --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
        --questions $Q --query-prefix="" --passage-prefix="" --max-seq-length 2048 `
        --out outputs/cache/ranks_public_v1.npz
    Assert-Ok "rank public v1"

    Write-Host "`n--- harrier ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank --name harrier `
        --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code `
        --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
        --questions $Q `
        --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
        --passage-prefix="" --max-seq-length 1024 `
        --out outputs/cache/ranks_public_harrier.npz
    Assert-Ok "rank public harrier"

    Write-Host "`n--- bm25 ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 --questions $Q `
        --out outputs/cache/ranks_public_bm25.npz
    Assert-Ok "rank public bm25"
}

Write-Host "`n=== [2/4] SINH 2 BAI NOP: baseline (beta=0) va prior (beta=$Beta) ===" -ForegroundColor Cyan
python -m src.evaluation.doctype_prior submit --beta $Beta
Assert-Ok "sinh bai nop"

Write-Host "`n=== [3/4] KIEM TRA DINH DANG ===" -ForegroundColor Cyan
python -m src.evaluation.validate_submission --predictions outputs/predictions/submission_baseline_regen.json
Assert-Ok "validate baseline"
python -m src.evaluation.validate_submission --predictions outputs/predictions/submission_doctype_prior.json
Assert-Ok "validate prior"

Write-Host "`n=== [4/4] DONG GOI 2 file zip ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Path "outputs/submissions" -Force | Out-Null
foreach ($pair in @(
    @{src = "outputs/predictions/submission_baseline_regen.json"; zip = "outputs/submissions/submission_v8_baseline_regen.zip" },
    @{src = "outputs/predictions/submission_doctype_prior.json"; zip = "outputs/submissions/submission_v9_doctype_prior.zip" }
)) {
    # LUU Y: KHONG dat ten bien nay la $stage - trung voi tham so -Stage o dau file (PowerShell
    # khong phan biet hoa thuong), gan gia tri ngoai ValidateSet se bi chan va script chet o day.
    $zipTmp = "outputs/predictions/_zip_tmp"
    if (Test-Path $zipTmp) { Remove-Item $zipTmp -Recurse -Force }
    New-Item -ItemType Directory -Path $zipTmp | Out-Null
    Copy-Item $pair.src (Join-Path $zipTmp "submission.json")
    if (Test-Path $pair.zip) { Remove-Item $pair.zip -Force }
    Compress-Archive -Path (Join-Path $zipTmp "submission.json") -DestinationPath $pair.zip
    Remove-Item $zipTmp -Recurse -Force
    Write-Host "  da tao $($pair.zip)" -ForegroundColor Green
}

Write-Host "`n=== CACH NOP VA DOC KET QUA ===" -ForegroundColor Yellow
Write-Host "NOP CA HAI, theo dung thu tu nay:" -ForegroundColor Yellow
Write-Host "  1. submission_v8_baseline_regen.zip   <- MOC MOI (thay cho con so 0.9208 da cu/khong tai lap duoc)" -ForegroundColor Yellow
Write-Host "  2. submission_v9_doctype_prior.zip    <- ban co prior" -ForegroundColor Yellow
Write-Host ""
Write-Host "Delta THAT = diem(v9) - diem(v8). CHI so sanh 2 con so nay voi nhau." -ForegroundColor Green
Write-Host "  Local du bao: +0.0072 (cuu 10 cau / hong 1 cau tren 1398 cau val, McNemar p=0.0117)." -ForegroundColor Green
Write-Host "  Public co 1000 cau -> ky vong khoang +5 den +9 cau doi tu sai sang dung." -ForegroundColor Green
Write-Host ""
Write-Host "NEU v8 KHAC HAN 0.9208: xac nhan pipeline da troi tu 21-08. Luc do MOI con so local" -ForegroundColor Red
Write-Host "  tu 22-08 tro di can doc lai theo moc v8, khong phai theo 0.9208." -ForegroundColor Red
