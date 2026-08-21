# Thu model embedding MOI (PRETRAINED, KHONG fine-tune) de thay AITeamVN/Vietnamese_Embedding -
# theo dung bai hoc quan trong nhat rut ra tu thi nghiem 2026-08-15: DOI SANG MODEL PRETRAINED
# MANH HON la huong hieu qua/it rui ro nhat da CHUNG MINH (bqbbao6 -> AITeamVN pretrained: +0.0586
# Score PUBLIC, KHONG can fine-tune gi). Fine-tune THEM tren AITeamVN da THAT BAI (0.8922 < 0.9058,
# xem submission-log-task1.md dong #12) - vi vay script nay CO CHU DICH KHONG fine-tune gi ca, chi
# doi model goc va danh gia LOCAL sach tren val_split_20 (khong ro ri, vi khong train tren du lieu
# nay).
#
# *** MODEL DUOC CHON: Alibaba-NLP/gte-Qwen2-1.5B-instruct *** (~1,78 ty tham so, van trong tran
# 4 ty). Chon thay vi Qwen/Qwen3-VL-Embedding-2B (dang xep uu tien hang dau trong allowed-models.md)
# vi ly do THUC TE cho 1 lan chay KHONG GIAM SAT qua dem, budget <8 tieng:
#   1. Tich hop CHUAN voi sentence-transformers (chi can --trust-remote-code) - Qwen3-VL la model
#      da phuong thuc (thiet ke chinh cho anh/tai lieu kieu ColPali), cach goi cho van ban thuan
#      CHUA duoc xac minh ky - rui ro loi giua chung cao hon, co the lang phi ca dem.
#   2. Co BANG CHUNG THAT tren tieng Viet: ban 7B cung ho gte-Qwen2 la model MANH NHAT trong 18
#      model duoc test tren VN-MTEB retrieval (46,05%) - Qwen3-VL-Embedding-2B moi, CHUA co
#      benchmark tieng Viet cong khai, chi co co so kien truc (RoPE) chu chua co so lieu thuc te.
#
# Script nay CHI encode + danh gia LOCAL (buoc ton thoi gian nhat, chay qua dem) - CHUA du doan
# public test / dong goi submission. Sang mai xem ket qua Recall LOCAL truoc, neu dang gia tri moi
# lam tiep: (1) sweep lai bm25-weight rieng cho model nay (giong sweep_bm25_weight_aiteamvn.ps1),
# (2) du doan public test + nop thu xac nhan.
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\run_pipeline_swap_gte_qwen2_overnight.ps1
# Neu gap loi CUDA out of memory: mo file nay, doi --batch-size 128 thanh --batch-size 64 (hoac
# thap hon) roi chay lai - cache .npz chi ghi khi encode xong TOAN BO nen lan chay lai se encode
# lai tu dau (khong resume duoc, khac voi finetune_embedding.py co checkpoint).

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

$Instruct = "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question."

# Buoc 0/2 - Sanity check NHANH (~30 giay - 1 phut): bat loi tich hop (trust_remote_code, prefix,
# max_seq_length) TRUOC khi cam ket ca dem vao buoc encode toan bo corpus o duoi. Neu buoc nay loi,
# DUNG LAI NGAY (khong sang buoc 1) - sua loi truoc, khong de no chay hong ca dem.
Invoke-Step "0/2 - Sanity check model gte-Qwen2-1.5B-instruct (nhanh, ~1 phut)" `
    "python scripts\sanity_check_embedding_model.py --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --trust-remote-code --max-seq-length 2048 --query-instruct=`"$Instruct`""

# Buoc 1/2 - Encode TOAN BO 351K chunk corpus (buoc TON THOI GIAN NHAT, uoc luong 4-8 tieng tuy
# GPU - day la phan "chay qua dem") + danh gia LOCAL SACH tren val_split_20 ngay sau khi encode
# xong (model nay la PRETRAINED, KHONG fine-tune tren du lieu cuoc thi nen KHONG rui ro ro ri du
# lieu - so sanh truc tiep CONG BANG voi AITeamVN pretrained duoc). --batch-size 128 (cao hon mac
# dinh 64) vi day la INFERENCE thuan tuy (khong luu gradient/optimizer state nhu luc train, it
# ton VRAM hon nhieu) - neu OOM, giam xuong 64 (xem ghi chu dau file). Cache .npz se duoc luu lai
# de tai su dung MIEN PHI cho sweep bm25-weight / du doan public test sau nay.
Invoke-Step "1/2 - Encode toan bo corpus + eval LOCAL tren val_split_20 (CHAY QUA DEM)" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model Alibaba-NLP/gte-Qwen2-1.5B-instruct --trust-remote-code --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/gte_qwen2_val_eval.json --embedding-cache outputs/cache/gte_qwen2_corpus.npz --bm25-weight 0.1 --query-instruct=`"$Instruct`" --passage-prefix=`"`" --max-seq-length 512 --batch-size 32" `
    "outputs/cache/gte_qwen2_corpus.npz"

Invoke-Step "2/2 - Cham diem Recall/Precision LOCAL (sach, khong ro ri)" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/gte_qwen2_val_eval.json --report outputs/predictions/gte_qwen2_val_eval_report.json"

Write-Host ""
Write-Host "=== HOAN TAT - so sanh Recall LOCAL vua do voi 2 baseline da biet: ===" -ForegroundColor Green
Write-Host "  - AITeamVN pretrained, bm25-weight=0.4: Recall LOCAL 0.9267 (dong #11, Score PUBLIC da xac nhan 0.9058)" -ForegroundColor Green
Write-Host "  - AITeamVN pretrained, bm25-weight=0.1: Recall LOCAL 0.9303 (CHUA xac nhan public)" -ForegroundColor Green
Write-Host ""
Write-Host "Neu gte-Qwen2 Recall LOCAL CAO HON RO RET (vd +0.01 tro len so voi 0.9303) - dang di tiep:" -ForegroundColor Yellow
Write-Host "  1. Sweep lai --bm25-weight RIENG cho model nay (bm25-weight toi uu co the khac AITeamVN -" -ForegroundColor Yellow
Write-Host "     copy sweep_bm25_weight_aiteamvn.ps1, doi --embedding-model/--embedding-cache sang gte_qwen2)." -ForegroundColor Yellow
Write-Host "  2. Sau khi chon duoc bm25-weight tot nhat, du doan tren public test + nop thu xac nhan." -ForegroundColor Yellow
Write-Host "Neu THAP HON hoac xap xi - giu nguyen AITeamVN pretrained, khong can di tiep huong nay." -ForegroundColor Yellow
