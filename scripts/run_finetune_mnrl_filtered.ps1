# =============================================================================
# A1 — FINE-TUNE LAN 4: MNRL + LOC FALSE NEGATIVE theo nguong cosine (2026-08-26)
#
# BOI CANH (xem submission_log_task1.md muc "LAN 3: fine-tune bang MNRL" de doc day du):
#   Lan 3 (MNRL, khong loc) KHONG sup do bieu dien nhung Recall -0.1080. Nguyen nhan da do duoc
#   TRUC TIEP (dung cache co san, khong ton GPU): ~12,3% hard-negative mined kieu cu co cosine >0.90
#   voi CHINH chunk dap an dung (gan trung noi dung, do 27 nhom tai lieu trung lap + GT thieu sot da
#   biet) -> day model rang doan van gan dung dap an la SAI, pha hong kha nang phan biet.
#
# LAN NAY (A1): GIU NGUYEN MNRL (khong margin, khong the sup do kieu Lan 2), CHI THEM 1 buoc LOC
#   truoc khi dua negative vao training: bo qua moi candidate co cosine(candidate, dap an dung) > 0.85.
#   Neu bi loc, quet SAU HON trong danh sach da xep hang (toi da 50 vi tri) de van du top-k negative
#   moi cau hoi, thay vi chi dung lai o vai candidate dau tien.
#
#   Ky thuat: dung LAI cache embedding corpus da co san (embcompare_aiteamvn.npz) de lay embedding
#   cua TUNG CANDIDATE NEGATIVE -> KHONG ton GPU (da encode san tu truoc). Chi encode THEM cac doan
#   'dap an dung' (positive) - re, vai nghin cau DUY NHAT, encode 1 lan truoc khi mining.
#
# ⚠️ 2 BUG DA SUA TRUOC KHI GIAO SCRIPT NAY (2026-08-26, xem finetune_embedding.py):
#   1. Them --hard-negative-cosine-max / --corpus-embedding-cache-for-filter bi loi argparse
#      (ky tu '%' trong help text bi hieu nham la format specifier) -> --help CRASH ngay tu dau,
#      nghia la MOI lenh goi deu se crash. Da sua (escape thanh '%%'), da verify lai --help chay
#      sach (khong chi py_compile - bai hoc cu tu bug --resume-from-checkpoint trung ten).
#   2. Phat hien THEM 1 flag CHET: --checkpoint-steps/--checkpoint-keep duoc khai bao + viet tai
#      lieu day du nhung KHONG dong nao doc args.checkpoint_steps/args.checkpoint_keep - co che
#      checkpoint THAT SU dang dung la --checkpoint-path/--checkpoint-save-steps/
#      --checkpoint-save-total-limit (da verify bang grep cheo). Da XOA 2 flag chet do de tranh
#      nham lan giong loi da gay mat checkpoint-250 lan truoc (2026-08-23).
#
# ⚠️ BAT BUOC: kiem tra checkpoint SOM (~12 phut, xem huong dan cuoi file). KHONG chay mu.
#
# Chay tu thu muc GOC:  .\scripts\run_finetune_mnrl_filtered.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

python -m src.training.finetune_embedding `
    --base-model AITeamVN/Vietnamese_Embedding `
    --train-json data/processed/train_split_80.json `
    --corpus data/raw/selected-contexts `
    --output-dir models/aiteamvn-ft-mnrl-filtered `
    --loss mnrl `
    --epochs 1 `
    --learning-rate 1e-5 `
    --batch-size 8 `
    --hard-negatives `
    --hard-negative-ranks outputs/cache/ranks_hold_v1.npz `
    --hard-negative-chunks data/processed/chunks_corpus.jsonl `
    --hard-negative-top-k 1 `
    --hard-negative-cosine-max 0.85 `
    --corpus-embedding-cache-for-filter outputs/cache/embcompare_aiteamvn.npz `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --fp16 `
    --checkpoint-path models/aiteamvn-ft-mnrl-filtered-ckpt `
    --checkpoint-save-steps 250 `
    --checkpoint-save-total-limit 5

Write-Host ""
Write-Host "=== XONG. KIEM TRA CHECKPOINT SOM truoc khi tin tuong ket qua cuoi ===" -ForegroundColor Green
Write-Host "  python scripts\diagnose_ft_model.py --model models\aiteamvn-ft-mnrl-filtered-ckpt\checkpoint-250" -ForegroundColor Green
Write-Host "  python scripts\quick_eval_finetune.py --new-model models\aiteamvn-ft-mnrl-filtered-ckpt\checkpoint-250 --top-n 50" -ForegroundColor Green
Write-Host ""
Write-Host "Sau khi train xong toan bo (~3-4 gio, it hon lan 3 vi mining bi loc bot ~12-21%% candidate):" -ForegroundColor Green
Write-Host "  python scripts\diagnose_ft_model.py --model models\aiteamvn-ft-mnrl-filtered" -ForegroundColor Green
Write-Host "  python scripts\quick_eval_finetune.py --new-model models\aiteamvn-ft-mnrl-filtered --top-n 50" -ForegroundColor Green
Write-Host ""
Write-Host "Neu quick_eval cho tin hieu DUONG (hoac it nhat KHONG am nhu Lan 3, -0.1080) -> dang encode" -ForegroundColor Yellow
Write-Host "lai TOAN BO corpus (~1.5-2h GPU) de do Recall LOCAL that tren val_split_20." -ForegroundColor Yellow
Write-Host "Neu van am ro ret -> DONG han huong fine-tune (4/4 lan that bai), khong thu lai nua." -ForegroundColor Yellow
