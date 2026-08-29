# =============================================================================
# FUSE THEO ĐIỂM (CombSUM) THAY VÌ CHỈ THEO HẠNG (RRF) — hướng #1 trong 3 hướng đã chốt thứ tự
# (fusion -> DEk21 -> sweep BM25 k1/b), 2026-08-28.
#
# VI SAO HUONG NAY: RRF hiện tại (`fuse` mặc định) CHỈ dùng vị trí xếp hạng (1, 2, 3...), KHÔNG BAO
#   GIỜ dùng độ lớn similarity/BM25 thật -> tài liệu hạng 1 với similarity 0.95 và hạng 1 với 0.40
#   đóng góp Y HỆT NHAU. Đây là thông tin bị vứt bỏ hoàn toàn từ trước tới nay, và đúng 1 trong 2
#   phạm trù DUY NHẤT từng đem lại cải thiện điểm thật (đổi CÁCH GỘP) — xem submission-log-task1.md.
#
# CƠ CHẾ: mỗi nguồn (v1, harrier, bm25) giờ LƯU THÊM điểm thô của đúng top-N ứng viên đã lưu hạng.
#   Khi fuse theo điểm: chuẩn hoá per-query (min-max, MỖI CÂU HỎI RIÊNG vì thang điểm khác hẳn nhau
#   giữa cosine similarity và BM25 không giới hạn) rồi CỘNG CÓ TRỌNG SỐ (CombSUM). Xem docstring đầy
#   đủ + code trong src/evaluation/ensemble_fusion.py (mục "THÊM 2026-08-28").
#
# TƯƠNG THÍCH NGƯỢC: `fuse` (không truyền --mode, hoặc --mode rrf) hoạt động Y HỆT trước giờ — code
#   nhánh RRF KHÔNG ĐỔI MỘT DÒNG. Regen ranks ở Stage 2 CHỈ THÊM key "scores" vào file .npz, các script
#   RRF cũ (run_bgem3_sparse.ps1, run_ensemble_fusion.ps1...) vẫn đọc được các file này bình thường.
#
# ĐÃ TEST TRƯỚC KHI GIAO (xem docstring ensemble_fusion.py, `selftest` + integration test riêng):
#   - save_ranks()/load_ranks() cũ đọc đúng file MỚI (bỏ qua key "scores" thừa) và file CŨ (không có
#     "scores") vẫn đọc bình thường -> 100% tương thích ngược.
#   - CombSUM đối chiếu tay với ví dụ nhỏ (đúng công thức score(d)=Σ w_i*norm_i(score_i(d))).
#   - Chạy qua CLI thật (không chỉ hàm nội bộ) trên dữ liệu tổng hợp: --mode rrf tái lập đúng
#     baseline, --mode score chạy đúng CombSUM, file thiếu scores bị --mode score TỪ CHỐI RÕ RÀNG
#     (không phải KeyError khó hiểu).
#
# CÁCH DÙNG (chạy từ thư mục GỐC repo):
#   .\scripts\run_score_fusion.ps1                    <- mặc định: SELFTEST (vài giây, không GPU)
#   .\scripts\run_score_fusion.ps1 -Stage regen-ranks  <- sinh lại ranks_v1/harrier/bm25 CÓ scores
#                                                          (dùng cache embedding có sẵn, KHÔNG encode
#                                                          lại corpus — chỉ encode 1398 câu hỏi, vài phút)
#   .\scripts\run_score_fusion.ps1 -Stage sweep        <- fuse --mode score, sweep trọng số
# =============================================================================
param(
    [ValidateSet("check", "regen-ranks", "sweep")]
    [string]$Stage = "check"
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
    Write-Host "`n=== SELFTEST: logic CombSUM + tuong thich nguoc (khong GPU, vai giay) ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion selftest
    Assert-Ok "selftest"
    Write-Host "`nBuoc tiep: .\scripts\run_score_fusion.ps1 -Stage regen-ranks" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "regen-ranks") {
    Write-Host "`n=== SINH LAI ranks_v1/harrier/bm25.npz VOI SCORES (dung cache embedding co san) ===" -ForegroundColor Cyan
    Write-Host "Chi encode LAI 1398 cau hoi (khong dung cache duoc vi query != passage), KHONG encode" -ForegroundColor DarkGray
    Write-Host "lai corpus (dung cache embcompare_*.npz co san) - vai phut, khong phai 1,5-2h nhu bge-m3." -ForegroundColor DarkGray

    Write-Host "`n--- v1 (AITeamVN/Vietnamese_Embedding) ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank --name v1 `
        --embedding-model AITeamVN/Vietnamese_Embedding `
        --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
        --out outputs/cache/ranks_v1.npz
    Assert-Ok "rank v1"

    Write-Host "`n--- harrier (mainguyen9/vietlegal-harrier-0.6b) ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank --name harrier `
        --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code `
        --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
        --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
        --max-seq-length 1024 --out outputs/cache/ranks_harrier.npz
    Assert-Ok "rank harrier"

    Write-Host "`n--- bm25 (khong can GPU/model) ---" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 --out outputs/cache/ranks_bm25.npz
    Assert-Ok "rank-bm25"

    Write-Host "`n=== XONG - ca 3 file .npz gio da co scores, VAN tuong thich voi moi script RRF cu ===" -ForegroundColor Green
    Write-Host "Buoc tiep:  .\scripts\run_score_fusion.ps1 -Stage sweep" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "sweep") {
    Write-Host "`n=== FUSE --mode score: giu v1=1.0 co dinh, SWEEP harrier + bm25 (minmax norm) ===" -ForegroundColor Cyan
    Write-Host "Luoi RONG vi CHUA BIET cua so trong so huu ich cua CombSUM (khac han RRF - dung song" -ForegroundColor DarkGray
    Write-Host "song voi luoi nay, KHONG suy dien tu cua so RRF da biet)." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion fuse --mode score --score-norm minmax `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25.npz `
        --weights harrier=0,0.25,0.5,0.75,1.0,1.5,2.0 `
        --weights bm25=0,0.1,0.25,0.5,1.0 `
        --report outputs/predictions/ensemble_score_fusion_report.json
    Assert-Ok "fuse --mode score"

    Write-Host "`n=== CACH DOC KET QUA ===" -ForegroundColor Yellow
    Write-Host "1. Dong 'harrier=0 bm25=0' la CombSUM chi dung v1 - vi minmax KHONG doi thu tu xep hang" -ForegroundColor Green
    Write-Host "   trong 1 nguon, dong nay PHAI ra DUNG Recall cua v1 DUNG RIENG (lich su du an: ~0,9267)." -ForegroundColor Green
    Write-Host "   Neu LECH nhieu -> co loi o dau do, BAO LAI, DUNG tin ket qua sweep." -ForegroundColor Red
    Write-Host "2. So sanh diem TOT NHAT cua sweep nay voi baseline RRF chinh thuc hien tai (0,9319)." -ForegroundColor Yellow
    Write-Host "   CombSUM la co che gop KHAC HAN RRF nen KHONG bat buoc phai bang 0,9319 o bat ky to hop" -ForegroundColor Yellow
    Write-Host "   nao - chi can so voi 0,9319 de biet CombSUM co TOT HON RRF hay khong." -ForegroundColor Yellow
    Write-Host "3. Quy tac tham dinh da chot (giong moi huong khac):" -ForegroundColor Yellow
    Write-Host "   >= +0.02  -> tin duoc, dang nop public." -ForegroundColor Green
    Write-Host "   +0.01..0.02 -> vung xam, can nhac (chi tin neu anh huong DEU len nhieu cau hoi)." -ForegroundColor Yellow
    Write-Host "   < +0.01   -> CHUA du bang chung, KHONG ket luan 'cai thien', dung dot luot nop." -ForegroundColor Red
    Write-Host "4. Neu ket qua tot nhat GAN baseline hoac THUA -> co the thu them --score-norm zscore" -ForegroundColor DarkGray
    Write-Host "   (it nhay voi 1-2 diem ngoai lai hon minmax) truoc khi ket luan dong huong." -ForegroundColor DarkGray
    exit 0
}
