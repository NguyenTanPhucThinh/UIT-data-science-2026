# Tao submission public cho AITeamVN pretrained (bm25=0.1, dong #11b, cau hinh chinh thuc) + entity-boost
# v4 injection (dong #15, --always-inject) - DUNG LAI cache corpus da encode san
# (outputs/cache/embcompare_aiteamvn.npz), KHONG encode lai 351K chunk corpus. Buoc 2 (entity-boost) cung
# dung lai CHINH cache nay de cham diem dense cho vai chuc cau hoi can xet boost - chi ton vai giay, khong
# dang ke so voi encode corpus.
#
# Vi sao KHONG do lai Recall LOCAL truoc khi nop lan nay: entity-boost da duoc do LOCAL day du va ghi log
# (dong #15, submission-log-task1.md) - Recall 0.9303 (AITeamVN khong boost) -> 0.9310 (co boost), THAT
# (khong phai nhieu, truy duoc dung 2 cau hoi duoc sua, KHONG case nao bi hai). Public test KHONG CO ground
# truth nen khong the "do LOCAL" tren no - script nay chay thang ra submission dua tren bang chung LOCAL da
# co san tren val_split_20.
#
# Cach dung (tu thu muc goc UIT-data-science-2026):
#   .\scripts\run_entity_boost_public.ps1
#
# CAP NHAT 2026-08-17: them --bm25-fast (da tu verify DUNG + AN TOAN bo nho tren chinh corpus that
# cua du an qua scripts/test_bm25_fast_on_real_corpus.py, --limit 50000: top-5 khop 100%, RAM dinh
# chi ~0.6GB cho 50K chunk) - giam buoc BM25 cua full_pipeline_eval tu ~30 phut xuong con ~35-40
# giay uoc tinh cho ca 1000 cau public test. Neu muon quay lai ban CU (khong --bm25-fast, an toan
# tuyet doi nhung cham), chi can xoa dong "--bm25-fast" ben duoi.

$ErrorActionPreference = "Stop"

Write-Host "=== BUOC 1/3: Tao hybrid predictions public test (AITeamVN pretrained, bm25=0.1 - dong #11b, --bm25-fast) ===" -ForegroundColor Cyan
python -m src.evaluation.full_pipeline_eval `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/public_test_preprocessed.jsonl `
    --output outputs/predictions/submission_entity_boost_public.json `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --stage-eval `
    --bm25-weight 0.1 `
    --query-prefix="" `
    --passage-prefix="" `
    --max-seq-length 2048 `
    --bm25-fast

Write-Host "`n=== BUOC 2/3: Ap dung entity-boost injection v4 (--always-inject) len ket qua hybrid ===" -ForegroundColor Cyan
Write-Host "    (dung file cau hoi public 'data/raw/public-official.json' chi de lay text cau hoi theo qid -" -ForegroundColor DarkGray
Write-Host "     ham load_ground_truth_questions() chi doc field 'question', khong dung field 'answer')" -ForegroundColor DarkGray
python scripts/apply_entity_boost_injection_v4.py `
    --always-inject `
    --ground-truth data/raw/public-official.json `
    --predictions outputs/predictions/submission_entity_boost_public.hybrid_fused.json `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --max-seq-length 2048 `
    --output outputs/predictions/submission_entity_boost_public_injected.json

Write-Host "`n=== BUOC 3/3: Dong goi submission.zip ===" -ForegroundColor Cyan
Copy-Item outputs\predictions\submission_entity_boost_public_injected.json outputs\predictions\submission.json -Force
Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_entity_boost.zip -Force

Write-Host "`n=== XONG ===" -ForegroundColor Green
Write-Host "Da tao: outputs\submissions\submission_entity_boost.zip"
Write-Host "Xem dong 'So cau hoi THUC SU bi sua doi' o BUOC 2 - ky vong ~1% tren 1000 cau public test (tuong" -ForegroundColor Yellow
Write-Host "tu 15/1398 = 1.1% da thay tren val_split_20)." -ForegroundColor Yellow
Write-Host "Nop file nay len he thong BTC, roi ghi lai Score PUBLIC vao submission-log-task1.md (dong moi," -ForegroundColor Yellow
Write-Host "so sanh voi 0.9173 = dong #11b KHONG entity-boost)." -ForegroundColor Yellow
