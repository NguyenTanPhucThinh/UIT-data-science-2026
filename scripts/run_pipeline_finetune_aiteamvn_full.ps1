# Fine-tune AITeamVN/Vietnamese_Embedding tren TOAN BO train.json (100%, KHONG chi 80%) - theo
# yeu cau cua Hoang (2026-08-15): bo qua Giai doan 1 (validate tren 80/20) vi ton thoi gian, di
# thang vao ban dung de nop that.
#
# *** DANH DOI QUAN TRONG CAN HIEU RO (2026-08-15) ***
# Vi train TREN CA val_split_20 (nam trong train.json day du), KHONG con cach nao danh gia LOCAL
# sach (khong ro ri) cho ban model nay nua - se KHONG biet duoc fine-tune co thuc su cai thien hon
# ban PRETRAINED (Recall LOCAL 0.9267, Score PUBLIC 0.9058, dong #11) hay khong cho toi khi NOP
# THU len BTC va so Score PUBLIC. Day la rui ro thuc su - khac voi bqbbao6 truoc day (fine-tune
# CHAC CHAN giup, 0.7806 -> 0.8624), voi AITeamVN CHUA co bang chung fine-tune se giup (model da
# duoc train tren 1.1M triplet noi bo, fine-tune them chi voi ~7-8 nghin cap tu data cuoc thi CO
# THE gay overfitting/catastrophic forgetting thay vi cai thien). Neu ban fine-tune ra ket qua te
# hon, van con cau hinh PRETRAINED da xac nhan public 0.9058 de dung lai (khong mat gi ngoai thoi
# gian GPU + 1 luot nop kiem tra) - xem submission-log-task1.md dong #11.
#
# Neu ban van muon co 1 tin hieu LOCAL (khong hoan toan sach nhung it nhat phat hien duoc loi
# nghiem trong, vd Recall tut xuong gan 0 vi hong model) truoc khi nop, co the tu chay them:
#   python -m src.evaluation.full_pipeline_eval --embedding-model models/vietnamese-embedding-aiteamvn-ft-full --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/aiteamvn_ft_full_sanity.json --embedding-cache outputs/cache/aiteamvn_ft_full_sanity.npz --bm25-weight 0.4 --query-prefix="" --passage-prefix="" --max-seq-length 2048
#   python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/aiteamvn_ft_full_sanity.json
# LUU Y: so nay se BI THOI PHONG (val nam trong tap train) - KHONG dung de so sanh cong bang voi
# 0.9267 cua ban pretrained, chi dung de bat loi ro rang (vd Recall < 0.5 la dau hieu hong).
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\run_pipeline_finetune_aiteamvn_full.ps1

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

# Buoc 1/4 - Fine-tune tren FULL train.json (~7000 cau, nhieu hon 80% split ~25%, nen cham hon
# tuong ung ~25% so voi lan chay 80% ban vua thu). --fp16 + checkpoint mac dinh moi 200 step (xem
# finetune_embedding.py, them 2026-08-15) - neu crash giua chung dung --resume-from-checkpoint.
Invoke-Step "1/4 - Fine-tune AITeamVN/Vietnamese_Embedding (FULL 100% train.json, fp16)" `
    "python -m src.training.finetune_embedding --base-model AITeamVN/Vietnamese_Embedding --train-json data/raw/train.json --corpus data/raw/selected-contexts --output-dir models/vietnamese-embedding-aiteamvn-ft-full --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048 --fp16 --epochs 2" `
    "models/vietnamese-embedding-aiteamvn-ft-full"

# Buoc 2/4 - Du doan tren public_test_preprocessed.jsonl (BAN DA TACH TU - KHONG dung file tho
# public-official.json, xem bug 2026-08-13 da ghi trong submission-log-task1.md). KHONG rerank,
# bm25-weight=0.4 (giong het cau hinh pretrained dang dung, chi doi embedding-model).
Invoke-Step "2/4 - Du doan tren public test (KHONG rerank, bm25-weight=0.4)" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model models/vietnamese-embedding-aiteamvn-ft-full --chunks data/processed/chunks_corpus.jsonl --questions data/processed/public_test_preprocessed.jsonl --output outputs/predictions/submission_v5_aiteamvn_ft.json --embedding-cache outputs/cache/aiteamvn_ft_full_public.npz --bm25-weight 0.4 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048"

# Buoc 3/4 - Kiem tra dinh dang truoc khi dong goi - BAT BUOC, tranh nop sai dinh dang mat luot.
Invoke-Step "3/4 - Kiem tra dinh dang submission" `
    "python -m src.evaluation.validate_submission --predictions outputs/predictions/submission_v5_aiteamvn_ft.json --corpus data/raw/selected-contexts --questions data/raw/public-official.json"

# Buoc 4/4 - Dong goi thanh submission.zip (BTC yeu cau ten file BEN TRONG zip la submission.json).
Invoke-Step "4/4 - Dong goi submission.zip" `
    "Copy-Item outputs\predictions\submission_v5_aiteamvn_ft.json outputs\predictions\submission.json -Force; Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_v5_aiteamvn_ft.zip -Force"

Write-Host ""
Write-Host "=== HOAN TAT - da co outputs\submissions\submission_v5_aiteamvn_ft.zip, san sang nop BTC ===" -ForegroundColor Green
Write-Host "Nop xong, so Score PUBLIC voi 0.9058 (ban PRETRAINED, dong #11 submission-log-task1.md):" -ForegroundColor Green
Write-Host "  - Neu CAO HON: fine-tune co ich, chinh thuc doi cau hinh dang dung sang ban nay." -ForegroundColor Green
Write-Host "  - Neu THAP HON hoac bang: fine-tune KHONG giup (co the do catastrophic forgetting/overfit tren it du lieu) - quay lai dung ban PRETRAINED, khong can fine-tune them cho AITeamVN nua." -ForegroundColor Yellow
