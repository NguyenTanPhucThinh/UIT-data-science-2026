# Thu lai reranker BAAI/bge-reranker-v2-m3 (PRETRAINED, KHONG fine-tune) tren chinh baseline
# AITeamVN/Vietnamese_Embedding dang dung chinh thuc (dong #11, Recall LOCAL 0.9267, Score PUBLIC
# 0.9058) - ket luan cu "reranker lam giam diem" (dong #9 stage-eval, xem submission-log-task1.md)
# duoc do tren baseline base bqbbao6 YEU HON nhieu (hybrid_fused chi 0.8791) - CHUA test lai voi
# baseline AITeamVN manh hon han. Dung lai cache .npz co san (khong ma hoa lai corpus) nen re va
# nhanh (~5-15 phut tuy GPU, phan lon thoi gian la chay CrossEncoder rerank tren 1398 cau hoi).
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\test_reranker_on_aiteamvn.ps1
#
# --stage-eval de co CA 2 so lieu de so sanh truc tiep:
#   - <output>.hybrid_fused.json = TRUOC rerank (nen ~ khop voi dong #11 da biet, 0.9267/0.1973)
#   - <output>.json (final)      = SAU rerank   (so lieu MOI, can biet)
# Neu final > hybrid_fused -> reranker THUC SU giup tren baseline nay, dang can nhac dua vao pipeline
# chinh thuc. Neu final < hybrid_fused (giong dong #9 cu) -> xac nhan lai ket luan cu, tiep tuc KHONG
# dung reranker.

python -m src.evaluation.full_pipeline_eval `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --reranker-model BAAI/bge-reranker-v2-m3 `
    --chunks data/processed/chunks_corpus.jsonl `
    --questions data/processed/val_preprocessed.jsonl `
    --output outputs/predictions/aiteamvn_rerank_test.json `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --stage-eval `
    --bm25-weight 0.4 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048

if ($LASTEXITCODE -ne 0) {
    Write-Host "LOI khi chay full_pipeline_eval - dung lai, khong cham diem." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Diem TRUOC rerank (hybrid_fused - nen ~ khop dong #11: 0.9267/0.1973) ===" -ForegroundColor Cyan
python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/aiteamvn_rerank_test.hybrid_fused.json --report outputs/predictions/aiteamvn_rerank_test_hybrid_fused_report.json

Write-Host ""
Write-Host "=== Diem SAU rerank (final - so lieu MOI can biet) ===" -ForegroundColor Cyan
python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/aiteamvn_rerank_test.json --report outputs/predictions/aiteamvn_rerank_test_report.json

Write-Host ""
Write-Host "So sanh 2 dong Recall/Precision in o tren: SAU > TRUOC -> reranker giup, nen can nhac dung." -ForegroundColor Yellow
Write-Host "SAU < TRUOC -> xac nhan lai ket luan cu, tiep tuc KHONG dung reranker trong pipeline chinh thuc." -ForegroundColor Yellow
