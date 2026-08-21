# Tao submission public voi mainguyen9/vietlegal-harrier-0.6b (PRETRAINED, KHONG fine-tune, KHONG rerank).
# DUNG LAI cache corpus da encode o buoc do LOCAL (embcompare_vietlegal_harrier.npz) - KHONG encode lai
# 351K chunk, chi encode them cau hoi public test (nhanh, vai phut). Neu CHUA chay buoc do LOCAL truoc
# (full_pipeline_eval.py tren val_preprocessed.jsonl), script nay se TU encode ca corpus luon (cham hon
# nhieu, lan dau) - cache sinh ra luc do van dung lai duoc cho lan sau.
#
# ⚠️ Chi nen chay script nay SAU KHI da xem qua Recall LOCAL va thay hop ly (>= muc dang dung hien tai,
# hoac it nhat khong thap han han) - dung nop mu theo dung nguyen tac "do LOCAL truoc" da ap dung xuyen
# suot du an. bm25-weight=0.1 o day la diem khoi dau (giong AITeamVN) - CHUA sweep rieng cho model nay,
# neu Recall LOCAL o 0.1 khong ro rang thi nen sweep lai (`sweep_bm25_weight.py --embedding-model
# mainguyen9/vietlegal-harrier-0.6b ...`) truoc khi quyet dinh nop.
#
# Cach dung (tu thu muc goc UIT-data-science-2026):
#   .\scripts\submit_vietlegal_harrier.ps1

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
    --max-seq-length 1024 `

# Lay file .hybrid_fused.json (= final vi KHONG dung reranker - dung format da chot cho AITeamVN)
Copy-Item outputs\predictions\submission_vietlegal_harrier.hybrid_fused.json outputs\predictions\submission.json -Force

# BAT BUOC: zip chi duoc chua duy nhat file "submission.json" (quy dinh BTC)
Compress-Archive -Path outputs\predictions\submission.json -DestinationPath outputs\submissions\submission_vietlegal_harrier.zip -Force

Write-Host ""
Write-Host "Da tao xong: outputs\submissions\submission_vietlegal_harrier.zip"
Write-Host "Nop file nay len he thong BTC, roi ghi lai Score PUBLIC vao submission-log-task1.md (dong moi)."
