# So sanh cach GOP DIEM tu chunk len document (aggregate_to_documents) - hien tai LUON dung max-score
# (chunk diem cao nhat/doc), CHUA TUNG thu cach khac trong toan bo log du an. Giu NGUYEN moi tham so
# khac o dinh da co bang chung (AITeamVN v1, dense-top-n=100, bm25-top-n=100, rrf-k=60, bm25-weight=0.1).
#
# DONG LUC: nhom cau hoi n_relevant>=2 (dang YEU NHAT trong moi lan do, Recall ~0.74-0.79) co the huong
# loi neu 1 document duoc cong dong diem tu NHIEU chunk lien quan thay vi chi lay 1 chunk manh nhat.
#
# Dung lai cache co san outputs/cache/embcompare_aiteamvn.npz - KHONG can encode lai corpus.
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\sweep_aggregation_aiteamvn.ps1

python -m src.evaluation.sweep_aggregation `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --dense-top-n 100 --bm25-top-n 100 --rrf-k 60 --bm25-weight 0.1 `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --report outputs/predictions/sweep_aggregation_report.json

Write-Host ""
Write-Host "=== Xong. So Recall cua sum_top2/sum_top3/mean_top2 voi 'max' (mac dinh hien tai, 0.9303 LOCAL) ===" -ForegroundColor Green
Write-Host "Neu cach nao cai thien RO RET (+0.005 tro len) - bao lai cho Claude de doi code + xac nhan lai" -ForegroundColor Green
Write-Host "tren public test truoc khi doi cau hinh chinh thuc." -ForegroundColor Green
