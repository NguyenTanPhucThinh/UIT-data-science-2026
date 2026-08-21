# Sweep dense-top-n / bm25-top-n / rrf-k (RRF fusion hyperparameter) cho AITeamVN/Vietnamese_Embedding
# v1 pretrained - cau hinh CHINH THUC dang dung (dong #11b, Score PUBLIC 0.9173).
#
# DONG LUC (2026-08-17): soat lai claude/submission-log-task1.md phat hien dong dau file ghi "tune
# RRF hyperparameter (dense-top-n/bm25-top-n/rrf-k) da DONG", nhung thuc ra chi co 1 lan nhac thoang
# qua "+0.0004" KHONG co bang sweep/dong log rieng - kiem tra code xac nhan 3 tham so nay GIU NGUYEN
# gia tri mac dinh (100/100/60) xuyen suot MOI lan chay ke tu dong #4 - tuc CHUA thuc su sweep. Script
# nay lap khoang trong do bang 1 luoi sweep that (mac dinh 3x3x4=36 to hop).
#
# GIU NGUYEN bm25-weight=0.1 (dinh da tim duoc rieng, dong #11b) - KHONG sweep lai dong thoi ca 4
# tham so (khong gian to hop qua lon).
#
# Dung lai cache co san outputs/cache/embcompare_aiteamvn.npz - KHONG can encode lai corpus.
# Dung --bm25-fast (luon bat trong script Python) de tinh BM25 nhanh cho ca 1398 cau hoi val.
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\sweep_rrf_hyperparams_aiteamvn.ps1

python -m src.evaluation.sweep_rrf_hyperparams `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --dense-top-n-grid 50,100,200 `
    --bm25-top-n-grid 50,100,200 `
    --rrf-k-grid 10,30,60,100 `
    --bm25-weight 0.1 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --report outputs/predictions/sweep_rrf_hyperparams_report.json

Write-Host ""
Write-Host "=== Xong. So Recall cao nhat trong bang tren voi moc mac dinh hien tai (100,100,60 -> 0.9303 LOCAL) ===" -ForegroundColor Green
Write-Host "Neu to hop khac cai thien RO RET (+0.005 tro len) - bao lai cho Claude de sweep tinh hon" -ForegroundColor Green
Write-Host "quanh to hop do, roi xac nhan lai tren public test truoc khi doi cau hinh chinh thuc." -ForegroundColor Green
