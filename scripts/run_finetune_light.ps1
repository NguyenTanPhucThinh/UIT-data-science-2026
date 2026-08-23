# =============================================================================
# FINE-TUNE EMBEDDING "NHE TAY" - AITeamVN/Vietnamese_Embedding  (2026-08-22)
#
# SUA DUNG 2 NGUYEN NHAN da chan doan cua that bai dong #12 (public 0.8922 < pretrained 0.9058):
#   1. LEARNING RATE: truoc day KHONG co tham so nay -> luon chay mac dinh 2e-5 cua
#      sentence-transformers. Voi model 568M da hoc 1,1 TRIEU triplet ma fine-tune tiep tren ~7-8
#      nghin cap, 2e-5 x 3 epoch = cong thuc catastrophic forgetting. Nay dung 5e-6 x 1 epoch.
#   2. HARD NEGATIVE YEU: ban cu lay mau NGAU NHIEN 2000/8532 van ban (23%) va chi dung CHUNK DAU
#      moi van ban. Nay lay tu THU HANG CUA RETRIEVER THAT (ranks_hold_v1.npz, top-200 cho dung
#      5602 cau train) -> negative chinh la loi that pipeline dang mac. KHONG ton them GPU.
#
# DO SACH KHONG TON LUOT NOP: train tren 5602 cau (train_split_80), do tren 1398 cau (val_split_20).
# Hai tap KHONG giao nhau -> khac han dong #12 (train 100% train.json nen val bi leak, phai dot 1
# luot nop moi biet la te).
#
# Chay tu thu muc GOC:  .\scripts\run_finetune_light.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== BUOC 0: tao lai train_split_80.json bang script chinh thuc cua du an ===" -ForegroundColor Cyan
python -m src.training.split_train_json `
    --train-json data/raw/train.json `
    --split-file data/processed/train_val_split.json `
    --output-dir data/processed

Write-Host "`n=== BUOC 1: fine-tune NHE TAY (1 epoch, lr 5e-6, hard negative tu retriever that) ===" -ForegroundColor Cyan
Write-Host "    Model se luu tai models/aiteamvn-ft-light" -ForegroundColor DarkGray
python -m src.training.finetune_embedding `
    --base-model AITeamVN/Vietnamese_Embedding `
    --train-json data/processed/train_split_80.json `
    --corpus data/raw/selected-contexts `
    --output-dir models/aiteamvn-ft-light `
    --epochs 1 `
    --learning-rate 5e-6 `
    --batch-size 4 `
    --hard-negatives `
    --hard-negative-ranks outputs/cache/ranks_hold_v1.npz `
    --hard-negative-chunks data/processed/chunks_corpus.jsonl `
    --hard-negative-top-k 3 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --fp16 `
    --checkpoint-path models/aiteamvn-ft-light-ckpt `
    --checkpoint-save-steps 500

Write-Host ""
Write-Host "=== XONG BUOC 1. DUNG LAI DE KIEM TRA TRUOC KHI TON THEM GPU ===" -ForegroundColor Green
Write-Host "Buoc tiep theo (encode lai toan bo corpus voi model moi) ton NHIEU GIO." -ForegroundColor Yellow
Write-Host "Bao lai cho Claude log training (loss dau/cuoi, so triplet) truoc khi chay buoc do." -ForegroundColor Yellow
