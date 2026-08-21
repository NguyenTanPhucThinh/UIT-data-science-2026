# Kiem tra hau-xu-ly dedup+backfill cho van de "tai lieu trung lap noi dung" (27 nhom/54 doc_id,
# phat hien 2026-08-17 - xem bao-cao-data-quality-cho-Kien.md muc 2). KHONG doi corpus/GT, KHONG doi
# model/hyperparameter - chi doi buoc lay top-5 CUOI CUNG: neu 2 doc_id trong top-K deu la cung 1 van
# ban (trung lap), bo qua 1 cai, lay ung vien xep hang thap hon de LAP DAY slot con lai.
#
# Dung lai cache co san outputs/cache/embcompare_aiteamvn.npz - KHONG can encode lai corpus.
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\eval_dedup_backfill_aiteamvn.ps1

python -m src.evaluation.eval_dedup_backfill `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --dup-groups data/processed/duplicate_doc_groups.json `
    --dense-top-n 100 --bm25-top-n 100 --rrf-k 60 --bm25-weight 0.1 `
    --candidate-pool 30 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --report outputs/predictions/eval_dedup_backfill_report.json

Write-Host ""
Write-Host "=== Xong. So Recall cua dedup_backfill voi baseline (max, khong dedup, 0.9303 LOCAL) ===" -ForegroundColor Green
Write-Host "Neu Recall tang RO RET (+0.001 tro len) hoac ngay ca khong doi - AN TOAN de ap dung vao" -ForegroundColor Green
Write-Host "pipeline chinh thuc (khong co rui ro giam diem). Bao lai cho Claude ket qua de quyet dinh." -ForegroundColor Green
