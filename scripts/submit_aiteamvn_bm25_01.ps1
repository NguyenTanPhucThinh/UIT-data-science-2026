# Tao submission MOI voi bm25-weight=0.1 (dinh moi tim duoc qua sweep 2026-08-15 toi, dong #11b,
# Recall LOCAL 0.9303) thay vi 0.4 dang dung (dong #11, Recall LOCAL 0.9267, Score PUBLIC 0.9058 da
# xac nhan). Day la lan nop CHINH de xac nhan bm25-weight=0.1 tren public test that - cai thien local
# +0.0036 nho hon nguong "+0.01" tu dat ra nen truoc do CHUA nop rieng, gio gop vao lan nop tiep theo
# nay.
#
# SUA LOI 2026-08-16: ban truoc cua script nay tro vao cache "embcompare_aiteamvn_public.npz" (CHUA
# TUNG duoc tao ra tren may nay) khien no ma hoa lai TOAN BO 351K chunk tu dau (~1.5-2 tieng lang phi).
# Cache .npz chi phu thuoc vao (model, corpus, prefix, max-seq-length) - KHONG phu thuoc vao tap cau
# hoi dang dung (val_preprocessed.jsonl hay public_test_preprocessed.jsonl deu dung chung 1 file
# chunks_corpus.jsonl) - nen dung LAI DUNG cache "embcompare_aiteamvn.npz" da xac nhan ton tai va
# hoat dong tot (dung cho sweep bm25-weight VA test reranker truoc do trong cung phien lam viec) la
# an toan va nhanh hon nhieu (chi mat vai phut, khong ma hoa lai).
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\submit_aiteamvn_bm25_01.ps1
#
# Sau khi chay xong, file outputs\submissions\submission_v6_aiteamvn_bm25p1.zip san sang de nop len
# he thong BTC (upload thu cong qua web, giong cac lan truoc).

python -m src.evaluation.full_pipeline_eval `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/public_test_preprocessed.jsonl `
    --output outputs/predictions/submission_v6_aiteamvn_bm25p1.json `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --stage-eval `
    --bm25-weight 0.1 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048

if ($LASTEXITCODE -ne 0) {
    Write-Host "LOI khi chay full_pipeline_eval - dung lai, khong tao submission." -ForegroundColor Red
    exit 1
}

# Lay file .hybrid_fused.json (= final vi KHONG dung reranker, dung cau hinh chinh thuc hien tai)
Copy-Item outputs\predictions\submission_v6_aiteamvn_bm25p1.hybrid_fused.json outputs\predictions\submission.json -Force
Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_v6_aiteamvn_bm25p1.zip -Force

Write-Host ""
Write-Host "=== XONG - da tao outputs\submissions\submission_v6_aiteamvn_bm25p1.zip ===" -ForegroundColor Green
Write-Host "Nop file nay len he thong BTC, ghi lai ID + Score PUBLIC vao submission-log-task1.md sau khi co ket qua." -ForegroundColor Green
