# =============================================================================
# SWEEP k1/b CỦA BM25Okapi — hướng #3 (cuối) trong 3 hướng đã chốt thứ tự
# (fusion theo score -> DEk21 -> sweep BM25 k1/b), 2026-08-28.
#
# VÌ SAO HƯỚNG NÀY, KHÁC `bm25-weight` ĐÃ SWEEP: `bm25-weight` chỉnh trọng số đóng góp của CẢ nhánh
#   BM25 vào RRF tổng (đã làm, +0.0115). k1/b chỉnh CHẤT LƯỢNG XẾP HẠNG NỘI BỘ của BM25 (chưa từng
#   sweep — luôn dùng mặc định thư viện k1=1.5/b=0.75, hiệu chỉnh cho corpus tiếng Anh TREC, không
#   có gì đảm bảo tối ưu cho corpus luật tiếng Việt đã tách từ hiện tại).
#
# ⚠️ RỦI RO PHA LOÃNG (biết trước): dù nội bộ tốt hơn bao nhiêu, BM25 vẫn chỉ đóng góp qua đúng
#   bm25-weight=0.1 hiện có -> cải thiện nội bộ nhỏ dễ bị hoà loãng, không đổi được top-5 cuối. Đây
#   là hướng RẺ NHẤT (thuần CPU) nhưng TRẦN CẢI THIỆN kỳ vọng cũng THẤP NHẤT trong 3 hướng.
#
# CƠ CHẾ RẺ: idf/doc_freqs/doc_len/avgdl KHÔNG phụ thuộc k1/b (xác minh trực tiếp trong
#   rank_bm25.BM25Okapi._calc_idf) — chỉ build ma trận tần suất thô 1 LẦN (`build-raw`, ~4GB RAM
#   đỉnh, như --bm25-fast gốc), mỗi combo (k1,b) sau đó CHỈ tính lại vector K rồi transform lại
#   .data của ma trận có sẵn — rẻ, không dựng lại rows/cols từ đầu.
#
# ĐÃ TEST TRƯỚC KHI GIAO (`selftest`, không cần corpus thật): đối chiếu TRỰC TIẾP (không suy luận
#   công thức đúng) với rank_bm25.BM25Okapi.get_scores() gốc, CẢ ở k1/b MẶC ĐỊNH lẫn k1/b TUỲ CHỈNH
#   khác mặc định — xác nhận "chỉ tính lại K" tương đương toán học với xây lại index từ đầu. Đã chạy
#   thêm end-to-end qua CLI thật (build-raw -> sweep -> export) trên dữ liệu tổng hợp, xác nhận file
#   export .npz tương thích ensemble_fusion.py (đọc được bằng load_ranks_scores()).
#
# CÁCH DÙNG (chạy từ thư mục GỐC repo):
#   .\scripts\run_bm25_k1b_sweep.ps1                    <- mặc định: SELFTEST (vài giây)
#   .\scripts\run_bm25_k1b_sweep.ps1 -Stage build-raw    <- build ma trận thô 1 lần (~4GB RAM đỉnh)
#   .\scripts\run_bm25_k1b_sweep.ps1 -Stage sweep        <- sweep lưới k1 x b (tức thì, thuần CPU)
#   .\scripts\run_bm25_k1b_sweep.ps1 -Stage ensemble-check <- đo Δ THẬT trên ensemble đầy đủ với
#                                                              combo tốt nhất (cần chỉnh -BestK1/-BestB
#                                                              theo kết quả sweep trước đó)
# =============================================================================
param(
    [ValidateSet("check", "build-raw", "sweep", "ensemble-check")]
    [string]$Stage = "check",
    [double]$BestK1 = 1.5,
    [double]$BestB = 0.75
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}

if ($Stage -eq "check") {
    Write-Host "`n=== SELFTEST: doi chieu TRUC TIEP voi rank_bm25.BM25Okapi.get_scores() goc ===" -ForegroundColor Cyan
    python -m src.evaluation.sweep_bm25_k1b selftest
    Assert-Ok "selftest"
    Write-Host "`nBuoc tiep: .\scripts\run_bm25_k1b_sweep.ps1 -Stage build-raw" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "build-raw") {
    Write-Host "`n=== BUILD ma tran tan suat tho (1 LAN, khong phu thuoc k1/b) ===" -ForegroundColor Cyan
    Write-Host "Canh bao RAM: uoc tinh dinh ~4GB cho corpus 351.394 chunk (giong --bm25-fast goc)." -ForegroundColor Yellow
    python -m src.evaluation.sweep_bm25_k1b build-raw --out outputs/cache/bm25_raw_tf.pkl
    Assert-Ok "build-raw"
    Write-Host "`nBuoc tiep:  .\scripts\run_bm25_k1b_sweep.ps1 -Stage sweep" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "sweep") {
    Write-Host "`n=== SWEEP luoi k1 x b (tuc thi, thuan CPU, khong dung lai ma tran tho) ===" -ForegroundColor Cyan
    python -m src.evaluation.sweep_bm25_k1b sweep --raw outputs/cache/bm25_raw_tf.pkl `
        --k1 1.0,1.2,1.5,1.8,2.2,2.6 --b 0.3,0.5,0.75,0.9 `
        --report outputs/predictions/bm25_k1b_sweep_report.json
    Assert-Ok "sweep"
    Write-Host "`n=== CACH DOC KET QUA ===" -ForegroundColor Yellow
    Write-Host "1. Day la Recall cua BM25 DUNG RIENG, KHONG PHAI ensemble cuoi cung - dung ket luan" -ForegroundColor Red
    Write-Host "   'cai thien' tu bang nay. Chi dung de CHON combo tot nhat de kiem tra tiep." -ForegroundColor Red
    Write-Host "2. Neu combo tot nhat KHAC xa mac dinh (k1=1.5,b=0.75) va Delta dung rieng lon (>0.02)" -ForegroundColor Yellow
    Write-Host "   -> dang thu -Stage ensemble-check voi -BestK1/-BestB tuong ung." -ForegroundColor Yellow
    Write-Host "3. Neu Delta dung rieng da RAT NHO (<0.005) -> gan nhu chac chan se bi pha loang het" -ForegroundColor DarkGray
    Write-Host "   o weight=0.1 khi vao ensemble - co the DONG huong ngay, khong can chay ensemble-check." -ForegroundColor DarkGray
    exit 0
}

if ($Stage -eq "ensemble-check") {
    Write-Host "`n=== Xuat ranks cho combo k1=$BestK1 b=$BestB, do Delta THAT tren ensemble day du ===" -ForegroundColor Cyan
    python -m src.evaluation.sweep_bm25_k1b export --raw outputs/cache/bm25_raw_tf.pkl `
        --k1 $BestK1 --b $BestB --out outputs/cache/ranks_bm25_tuned.npz
    Assert-Ok "export"

    Write-Host "`n=== FUSE: thay ranks_bm25.npz mac dinh bang ranks_bm25_tuned.npz, GIU NGUYEN bm25-weight=0.1 ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25_tuned.npz `
        --weights harrier=0.2 --weights bm25=0.1 --rrf-k 60 `
        --report outputs/predictions/ensemble_bm25_tuned_report.json
    Assert-Ok "fuse ensemble"

    Write-Host "`n=== CACH DOC KET QUA ===" -ForegroundColor Yellow
    Write-Host "So Recall dong DUY NHAT nay voi baseline chinh thuc 0.9319 (dung ranks_bm25.npz mac" -ForegroundColor Yellow
    Write-Host "dinh). Quy tac tham dinh nhu moi huong khac: >=+0.02 tin duoc; +0.01-0.02 vung xam;" -ForegroundColor Yellow
    Write-Host "<+0.01 CHUA du bang chung, dung dot luot nop." -ForegroundColor Yellow
    exit 0
}
