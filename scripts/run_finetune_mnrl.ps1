# =============================================================================
# FINE-TUNE LAN 2 — doi sang MultipleNegativesRankingLoss  (2026-08-23)
#
# LAN 1 THAT BAI (9 gio GPU): TripletLoss margin=5 tren embedding DA CHUAN HOA L2.
#   Khoang cach Euclid toi da chi = 2 < 5 -> margin KHONG BAO GIO thoa man -> gradient khong bao gio
#   tat -> model bi day toi loi giai THOAI HOA: nen ca khong gian 1024 chieu xuong 1 TRUC (moi vector
#   chi con +v hoac -v). Loss van giam deu rat dep suot 9 tieng, nhung Recall 0.92 -> 0.0046.
#   Trieu chung: cos(a,b) ra DUNG +-1.0000, cos trung binh voi van ban ngau nhien = -0.7578.
#
# LAN NAY:
#   --loss mnrl : softmax tren cosine, dung hard negative TUONG MINH + in-batch negatives.
#                 KHONG co margin -> KHONG THE thoai hoa kieu tren. Day la loss CHUAN cho retriever.
#   --hard-negative-top-k 1 : MNRL tu sinh them in-batch negative nen 1 negative kho la du.
#                 6.108 bo thay vi 18.324 -> ~3 tieng thay vi 9.
#   --learning-rate 1e-5 : nhinh hon 5e-6 lan truoc vi it buoc hon (764 thay vi 4581), nhung van
#                 duoi 2e-5 von da gay catastrophic forgetting o dong #12.
#
# ⚠️ BAT BUOC: kiem tra checkpoint SOM. Xem huong dan cuoi file. KHONG chay mu 3 tieng nua.
#
# Chay tu thu muc GOC:  .\scripts\run_finetune_mnrl.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

python -m src.training.finetune_embedding `
    --base-model AITeamVN/Vietnamese_Embedding `
    --train-json data/processed/train_split_80.json `
    --corpus data/raw/selected-contexts `
    --output-dir models/aiteamvn-ft-mnrl `
    --loss mnrl `
    --epochs 1 `
    --learning-rate 1e-5 `
    --batch-size 8 `
    --hard-negatives `
    --hard-negative-ranks outputs/cache/ranks_hold_v1.npz `
    --hard-negative-chunks data/processed/chunks_corpus.jsonl `
    --hard-negative-top-k 1 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --fp16 `
    --checkpoint-path models/aiteamvn-ft-mnrl-ckpt `
    --checkpoint-save-steps 250

Write-Host ""
Write-Host "=== XONG. Kiem tra model cuoi truoc khi encode toan corpus ===" -ForegroundColor Green
Write-Host "  python scripts\diagnose_ft_model.py --model models/aiteamvn-ft-mnrl" -ForegroundColor Green
Write-Host "  python scripts\quick_eval_finetune.py --new-model models/aiteamvn-ft-mnrl --top-n 50" -ForegroundColor Green
