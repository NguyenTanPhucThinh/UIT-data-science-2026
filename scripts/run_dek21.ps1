# =============================================================================
# NHÁNH `huyydangg/DEk21_hcmute_embedding` LÀM NGUỒN THỨ 4/5 CHO ENSEMBLE — hướng #2 trong 3 hướng đã
# chốt thứ tự (fusion -> DEk21 -> sweep BM25 k1/b), 2026-08-28.
#
# VI SAO HUONG NAY: base `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT/RoBERTa family),
#   ~0,1 tỷ tham số, train sẵn trên ~100K cặp QA pháp lý tiếng Việt, ndcg@10=0,918 trên VMTEB legal —
#   KIẾN TRÚC KHÁC HẲN v1/harrier hiện có (không phải biến thể). Cùng phạm trù đã từng đem lại cải
#   thiện thật (đổi/thêm TÍN HIỆU mới, không phải thêm tầng xử lý hay train lại).
#
# ⚠️ CẢNH BÁO QUAN TRỌNG NHẤT — max_seq_length=256 LÀ TRẦN KIẾN TRÚC CỨNG (RoBERTa dùng absolute
#   position embedding, KHÔNG như gte-Qwen2's 512 chỉ là lựa chọn CÓ THỂ chỉnh). Nếu kết quả kém,
#   đây gần như chắc chắn là NGUYÊN NHÂN CUỐI CÙNG, KHÔNG có "thử lại ở độ dài khác" như đã làm với
#   gte-Qwen2 trước đây. Một kết quả âm ở đây coi như ĐÓNG HẲN, không phải "tạm đóng".
#
# BẮT BUỘC input ĐÃ TÁCH TỪ (word-segmented, nối bằng "_") — model PhoBERT-family, XÁC MINH qua
#   model card (pyvi.ViTokenizer). Corpus/câu hỏi ĐÃ CÓ SẴN field song song passage_segmented/
#   question_segmented (không cần tách từ lại) — dùng cờ MỚI `--use-segmented-text` vừa thêm vào
#   `ensemble_fusion.py rank` (xem docstring "THÊM 2026-08-28" trong file đó). KHÔNG cần prefix/
#   instruction template (model card không yêu cầu), pooling mean + cosine similarity đã có sẵn
#   trong modules.json của model, không cần cấu hình thêm.
#
# ĐÃ TEST TRƯỚC KHI GIAO (script test riêng, dùng stub sentence_transformers — không cần GPU):
#   - --use-segmented-text đúng dùng passage_segmented/question_segmented khi có, fallback về bản
#     thô (không crash) khi thiếu field, có cảnh báo rõ ràng.
#   - KHÔNG bật cờ này (mặc định) -> luồng encode CŨ không đổi một dòng hành vi (v1/harrier vẫn chạy
#     y hệt trước).
#
# CÁCH DÙNG (chạy từ thư mục GỐC repo):
#   .\scripts\run_dek21.ps1                  <- mặc định: chỉ ENCODE + đo Recall ĐỨNG RIÊNG (~20-40 phút GPU)
#   .\scripts\run_dek21.ps1 -Stage fuse       <- fuse vào ensemble v1+harrier+bm25, sweep trọng số
# =============================================================================
param(
    [ValidateSet("standalone", "fuse")]
    [string]$Stage = "standalone",
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}

if ($Stage -eq "standalone") {
    Write-Host "`n=== [1/2] ENCODE corpus + cau hoi bang DEk21 (segmented text, max_seq_length=256) ===" -ForegroundColor Cyan
    Write-Host "Model nho (~0,1 ty) + sequence ngan (256 token thay vi 2048) -> nhanh hon nhieu so voi" -ForegroundColor DarkGray
    Write-Host "v1/harrier, uoc luong ~20-40 phut GPU cho ca corpus 351.394 chunk." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion rank --name dek21 `
        --embedding-model huyydangg/DEk21_hcmute_embedding `
        --embedding-cache outputs/cache/embcompare_dek21.npz `
        --allow-encode --use-segmented-text --max-seq-length 256 `
        --batch-size $BatchSize `
        --out outputs/cache/ranks_dek21.npz
    Assert-Ok "rank dek21"

    Write-Host "`n=== [2/2] Recall@5 ĐỨNG RIÊNG (RRF 1 nguồn = giữ nguyên thứ tự xếp hạng gốc) ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse `
        --rank dek21=outputs/cache/ranks_dek21.npz `
        --weights dek21=1.0 --rrf-k 60 `
        --report outputs/predictions/dek21_standalone_report.json
    Assert-Ok "fuse standalone"

    Write-Host "`n=== CÁCH ĐỌC KẾT QUẢ ===" -ForegroundColor Yellow
    Write-Host "So Recall dong 'dek21=1' voi cac model dung rieng da biet: v1 ~0,9267 · harrier 0,9040" -ForegroundColor Yellow
    Write-Host "· bge-m3 goc 0,8847 · gte-Qwen2 0,8864 (v2 0,9131)." -ForegroundColor Yellow
    Write-Host "  Gan hoac tren nhom nay (>=0,85) -> DANG thu lam nhanh phu, sang buoc fuse." -ForegroundColor Green
    Write-Host "  THAP HAN HAN (<0,7) -> rat co the la truncation 256 token that su cat mat noi dung" -ForegroundColor Red
    Write-Host "  can thiet -> DONG HAN huong nay (KHONG co 'thu lai o do dai khac' nhu gte-Qwen2)." -ForegroundColor Red
    Write-Host "`nBuoc tiep (neu Recall on):  .\scripts\run_dek21.ps1 -Stage fuse" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "fuse") {
    Write-Host "`n=== FUSE: giu v1=1.0/harrier=0.2/bm25=0.1/rrf_k=60, SWEEP rieng trong so dek21 ===" -ForegroundColor Cyan
    Write-Host "Luoi RONG (0 - 1,5) vi day la kien truc KHAC HAN v1/harrier (giong tinh huong harrier" -ForegroundColor DarkGray
    Write-Host "luc dau - cua so co the can trong so lon ~0,6-1,2 de 'mo rong pool', KHONG giong bm25" -ForegroundColor DarkGray
    Write-Host "hay bge-m3 sparse chi can trong so nho vi trung lap voi tin hieu co san)." -ForegroundColor DarkGray
    python -m src.evaluation.ensemble_fusion fuse `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25.npz `
        --rank dek21=outputs/cache/ranks_dek21.npz `
        --weights harrier=0.2 `
        --weights bm25=0.1 `
        --weights dek21=0,0.1,0.3,0.6,0.9,1.2,1.5 `
        --rrf-k 60 `
        --report outputs/predictions/ensemble_add_dek21_report.json
    Assert-Ok "fuse ensemble"

    Write-Host "`n=== CÁCH ĐỌC KẾT QUẢ ===" -ForegroundColor Yellow
    Write-Host "1. Dong 'dek21=0' PHAI ra DUNG 0,9319 (= cau hinh chinh thuc hien tai)." -ForegroundColor Green
    Write-Host "   Neu KHONG dung -> co loi o dau do, BAO LAI, DUNG tin bang ket qua." -ForegroundColor Red
    Write-Host "2. Quy tac tham dinh da chot: >= +0.02 tin duoc/dang nop; +0.01-0.02 vung xam (chi tin" -ForegroundColor Yellow
    Write-Host "   neu anh huong DEU len nhieu cau hoi); < +0.01 CHUA du bang chung, dung dot luot nop." -ForegroundColor Yellow
    Write-Host "3. Neu diem cao nhat GAN dinh cua luoi (vd 1.5) -> QUET THEM diem cao hon, cua so co the" -ForegroundColor DarkGray
    Write-Host "   chua het (giong bai hoc harrier ban dau)." -ForegroundColor DarkGray
    Write-Host "4. Neu giam DON DIEU tu trong so nho nhat -> DONG huong. Vi max_seq_length=256 la TRAN" -ForegroundColor Red
    Write-Host "   KIEN TRUC CUNG, ket luan DONG o day la CUOI CUNG, khong co huong quay lai." -ForegroundColor Red
    exit 0
}
