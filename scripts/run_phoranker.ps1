# =============================================================================
# PHORANKER + FINE-TUNE SEMI-HARD NEGATIVE — hướng cuối cùng còn lại (2026-08-28)
#
# BỐI CẢNH: mọi hướng khác đã đóng bằng bằng chứng PUBLIC (6 bài nộp ngày 28-08, cấu hình tốt nhất
#   vẫn là RRF baseline 0.9208). Đây là thứ DUY NHẤT còn lại có đúng bậc độ lớn cần thiết:
#   chẩn đoán cho thấy 85.7% lỗi còn lại là lỗi XẾP HẠNG với đáp án ĐÃ nằm trong pool
#   (Recall@100 = 0.9890 vs Recall@5 = 0.9319) -> +0.057 dư địa mà CHỈ rerank chạm tới được.
#
# KHÁC GÌ 4 PHÉP THỬ RERANKER CŨ (đều âm): model PhoRanker chuyên tiếng Việt thay vì bge đa ngữ ·
#   FINE-TUNE bằng SEMI-HARD negative thay vì dùng thẳng bản pretrained · TRỘN điểm có trọng số
#   thay vì thay thế thứ hạng. Bài arXiv 2507.14619 đo: semi-hard 78.21 MRR vs hard 47.96 vs
#   easy 56.75 — và 4 lần fine-tune thất bại của team đều dùng HARD negative.
#
# ⚠️ RỦI RO LỚN NHẤT, ĐO TRƯỚC KHI TỐN GPU: PhoRanker có max_length=256 (họ PhoBERT/RoBERTa, trần
#   kiến trúc CỨNG — đúng lý do đã đóng hẳn DEk21). Chunk của ta tới 2000 ký tự nên bị CẮT nhiều.
#   Vì vậy giai đoạn 3-4 (chấm bằng bản PRETRAINED) là CỬA CHẶN: nếu bản pretrained làm điểm giảm
#   đơn điệu ngay từ alpha nhỏ nhất thì gần như chắc chắn truncation đã giết tín hiệu -> DỪNG,
#   KHÔNG tốn GPU fine-tune.
#
# THỨ TỰ CHẠY:
#   .\scripts\run_phoranker.ps1                        <- 1. selftest cả 2 script (vài giây)
#   .\scripts\run_phoranker.ps1 -Phase preview          <- 2. xem cặp train bằng mắt (không GPU)
#   .\scripts\run_phoranker.ps1 -Phase score-pretrained <- 3. GPU: chấm val bằng PhoRanker gốc
#   .\scripts\run_phoranker.ps1 -Phase blend-pretrained <- 4. CỬA CHẶN: sweep alpha, quyết đi tiếp?
#   .\scripts\run_phoranker.ps1 -Phase train            <- 5. GPU: fine-tune semi-hard
#   .\scripts\run_phoranker.ps1 -Phase score-finetuned  <- 6. GPU: chấm lại val bằng bản đã train
#   .\scripts\run_phoranker.ps1 -Phase blend-finetuned  <- 7. sweep alpha lần cuối
# =============================================================================
param(
    [ValidateSet("check", "preview", "score-pretrained", "blend-pretrained", "train",
                 "score-finetuned", "blend-finetuned")]
    [string]$Phase = "check",
    [int]$TopNDocs = 30,
    [int]$Epochs = 2,
    [int]$BatchSize = 16,
    [string]$NegStrategy = "semi-hard"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$FtDir = "models/phoranker_semihard"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}

if ($Phase -eq "check") {
    Write-Host "`n=== SELFTEST ca 2 script (khong GPU) ===" -ForegroundColor Cyan
    python -m src.evaluation.rerank_blend selftest
    Assert-Ok "selftest rerank_blend"
    python -m src.training.finetune_phoranker selftest
    Assert-Ok "selftest finetune_phoranker"
    Write-Host "`nBuoc tiep: .\scripts\run_phoranker.ps1 -Phase preview" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "preview") {
    Write-Host "`n=== XEM CAP HUAN LUYEN BANG MAT (khong GPU, khong train) ===" -ForegroundColor Cyan
    python -m src.training.finetune_phoranker preview --negative-strategy $NegStrategy `
        --out outputs/predictions/phoranker_train_preview.jsonl
    Assert-Ok "preview"
    Write-Host "`nMO FILE outputs/predictions/phoranker_train_preview.jsonl VA DOC vai dong:" -ForegroundColor Yellow
    Write-Host "  - label=1.0 phai la doan van THUC SU tra loi cau hoi." -ForegroundColor Yellow
    Write-Host "  - label=0.0 phai la van ban KHAC nhung CUNG CHU DE (do lay tu pool ung vien)." -ForegroundColor Yellow
    Write-Host "  - Neu negative toan thu chang lien quan gi -> pool sai, BAO LAI truoc khi train." -ForegroundColor Red
    Write-Host "`nBuoc tiep: .\scripts\run_phoranker.ps1 -Phase score-pretrained" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "score-pretrained") {
    Write-Host "`n=== [GPU] Cham val bang PhoRanker GOC (chua fine-tune) ===" -ForegroundColor Cyan
    Write-Host "1398 cau x $TopNDocs document = ~$(1398*$TopNDocs) cap cross-encoder." -ForegroundColor DarkGray
    Write-Host "Day la phep thu RE de kiem tra rui ro truncation 256 token TRUOC khi ton GPU train." -ForegroundColor DarkGray
    python -m src.evaluation.rerank_blend score `
        --questions data/processed/val_preprocessed.jsonl `
        --reranker-model itdainb/PhoRanker `
        --top-n-docs $TopNDocs `
        --out outputs/cache/rerank_val_pretrained.npz
    Assert-Ok "score pretrained"
    Write-Host "`nBuoc tiep: .\scripts\run_phoranker.ps1 -Phase blend-pretrained" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "blend-pretrained") {
    Write-Host "`n=== CUA CHAN: sweep alpha voi PhoRanker GOC ===" -ForegroundColor Cyan
    python -m src.evaluation.rerank_blend blend `
        --rerank outputs/cache/rerank_val_pretrained.npz `
        --report outputs/predictions/rerank_pretrained_report.json
    Assert-Ok "blend pretrained"
    Write-Host "`n=== QUYET DINH DI TIEP HAY DUNG ===" -ForegroundColor Yellow
    Write-Host "Dong alpha=0 PHAI ra dung 0.9319. Neu khong -> co loi, BAO LAI." -ForegroundColor Green
    Write-Host ""
    Write-Host "GIAM DON DIEU ngay tu alpha nho nhat  -> gan nhu chac chan truncation 256 token da" -ForegroundColor Red
    Write-Host "  giet tin hieu (giong het DEk21). DUNG LAI, KHONG ton GPU fine-tune. Dong huong." -ForegroundColor Red
    Write-Host "CO dinh o alpha nao do (du nho)       -> reranker CO tin hieu that; fine-tune semi-hard" -ForegroundColor Green
    Write-Host "  co co so de khuech dai no. DANG chay tiep -Phase train." -ForegroundColor Green
    Write-Host ""
    Write-Host "LUU Y: o buoc nay ta CHUA quan tam Delta co du lon de nop hay khong — chi can biet" -ForegroundColor DarkGray
    Write-Host "co TIN HIEU hay khong, de quyet dinh co dau tu GPU fine-tune." -ForegroundColor DarkGray
    exit 0
}

if ($Phase -eq "train") {
    Write-Host "`n=== [GPU] FINE-TUNE PhoRanker voi negative = $NegStrategy ===" -ForegroundColor Cyan
    Write-Host "Chi dung 5602 cau train_split_80 (ranks_hold_*.npz). Script co CHOT CHAN cung:" -ForegroundColor DarkGray
    Write-Host "tu choi chay neu phat hien bat ky cau nao thuoc val_split_20." -ForegroundColor DarkGray
    python -m src.training.finetune_phoranker train `
        --negative-strategy $NegStrategy `
        --epochs $Epochs --batch-size $BatchSize `
        --output-dir $FtDir
    Assert-Ok "train"
    Write-Host "`nBuoc tiep: .\scripts\run_phoranker.ps1 -Phase score-finetuned" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "score-finetuned") {
    Write-Host "`n=== [GPU] Cham lai val bang ban DA FINE-TUNE ===" -ForegroundColor Cyan
    python -m src.evaluation.rerank_blend score `
        --questions data/processed/val_preprocessed.jsonl `
        --reranker-model $FtDir `
        --top-n-docs $TopNDocs `
        --out outputs/cache/rerank_val_finetuned.npz
    Assert-Ok "score finetuned"
    Write-Host "`nBuoc tiep: .\scripts\run_phoranker.ps1 -Phase blend-finetuned" -ForegroundColor Green
    exit 0
}

if ($Phase -eq "blend-finetuned") {
    Write-Host "`n=== Sweep alpha voi ban DA FINE-TUNE ===" -ForegroundColor Cyan
    python -m src.evaluation.rerank_blend blend `
        --rerank outputs/cache/rerank_val_finetuned.npz `
        --report outputs/predictions/rerank_finetuned_report.json
    Assert-Ok "blend finetuned"
    Write-Host "`n=== CACH DOC (quy tac da chot 28-08) ===" -ForegroundColor Yellow
    Write-Host "1. alpha=0 PHAI ra dung 0.9319." -ForegroundColor Green
    Write-Host "2. So sanh dinh voi 0.9319. Nguong dang nop: Delta >= 0.02 (KHONG phai 0.01) —" -ForegroundColor Yellow
    Write-Host "   3/3 lan Delta local duong duoi 0.01 deu DOI DAU tren public." -ForegroundColor Yellow
    Write-Host "3. DUNG chon alpha bang argmax cua sweep. Chon THAN TRONG hon dinh (bai hoc #21:" -ForegroundColor Red
    Write-Host "   harrier=0.2 chon nhe hon local-argmax la cau hinh duy nhat den gio van dung vung)." -ForegroundColor Red
    Write-Host "4. Neu Delta >= 0.02: sinh ranks public roi nop. Nho quy dinh muc 7 — them model moi" -ForegroundColor Yellow
    Write-Host "   phai gui de xuat BTC truoc 10 ngay so voi han chot private test." -ForegroundColor Yellow
    exit 0
}
