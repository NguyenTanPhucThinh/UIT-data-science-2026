# Gop 3 buoc (do LOCAL -> in Recall -> tao submission) thanh 1 script CHAY TUAN TU, khong can quay lai
# chat giua chung. LUU Y: khong chay 3 buoc THAT SU DONG THOI (song song) - vi buoc 3 (submission)
# DUNG LAI cache corpus sinh ra o buoc 1 (embcompare_vietlegal_harrier.npz), chi encode THEM cau hoi
# public test (nhanh) - neu chay song song thi 2 tien trinh se cung co gang encode 351K chunk corpus
# 1 luc, vua lang phi GPU/VRAM vua co the OOM, ma buoc 3 van phai doi buoc 1 xong moi co cache dung de
# dung lai. Chay TUAN TU (nhu duoi day) la cach nhanh nhat THAT SU (khong ton VRAM/GPU cho 2 lan encode
# corpus giong het nhau).
#
# Ban van co the xem Recall LOCAL in ra o buoc 2 TRUOC KHI thuc su nop file .zip len he thong BTC (buoc
# up load la thao tac tay tren web, khong nam trong script nay) - neu Recall thap hon han AITeamVN
# (0.9303), dung nop file .zip sinh ra, chi giu lai de tham khao.
#
# Cach dung (tu thu muc goc UIT-data-science-2026):
#   .\scripts\run_vietlegal_harrier_full.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== BUOC 1/3: Do LOCAL (encode toan bo corpus + val_split_20) ===" -ForegroundColor Cyan
python -m src.evaluation.full_pipeline_eval `
    --embedding-model mainguyen9/vietlegal-harrier-0.6b `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/val_preprocessed.jsonl `
    --output outputs/predictions/vietlegal_harrier_val_eval.json `
    --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
    --stage-eval `
    --bm25-weight 0.1 `
    --trust-remote-code `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" `
    --max-seq-length 1024

Write-Host "`n=== BUOC 2/3: In Recall LOCAL (doc ky truoc khi quyet dinh nop) ===" -ForegroundColor Cyan
python -m src.evaluation.evaluate `
    --ground-truth data/processed/val_split_20.json `
    --predictions outputs/predictions/vietlegal_harrier_val_eval.hybrid_fused.json

Write-Host "`n=== BUOC 3/3: Tao submission (DUNG LAI cache buoc 1, chi encode them cau hoi public) ===" -ForegroundColor Cyan
python -m src.evaluation.full_pipeline_eval `
    --embedding-model mainguyen9/vietlegal-harrier-0.6b `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/public_test_preprocessed.jsonl `
    --output outputs/predictions/submission_vietlegal_harrier.json `
    --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
    --stage-eval `
    --bm25-weight 0.1 `
    --trust-remote-code `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" `
    --max-seq-length 1024

Copy-Item outputs\predictions\submission_vietlegal_harrier.hybrid_fused.json outputs\predictions\submission.json -Force
Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_vietlegal_harrier.zip -Force

Write-Host "`n=== XONG ===" -ForegroundColor Green
Write-Host "Da tao: outputs\submissions\submission_vietlegal_harrier.zip"
Write-Host "⚠️  XEM LAI dong 'Recall trung binh' o BUOC 2 phia tren TRUOC KHI nop file .zip nay len he" -ForegroundColor Yellow
Write-Host "    thong BTC. Neu THAP HON han 0.9303 (AITeamVN dang dung), KHONG nop - chi giu tham khao." -ForegroundColor Yellow
