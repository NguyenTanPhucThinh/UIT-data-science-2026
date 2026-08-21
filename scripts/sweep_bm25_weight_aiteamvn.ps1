# Sweep --bm25-weight rieng cho AITeamVN/Vietnamese_Embedding PRETRAINED (dong #11,
# submission-log-task1.md) - cau hinh dang dung CHINH THUC (Score PUBLIC 0.9058, ID 889673).
#
# *** CAP NHAT (2026-08-15 toi) *** Ban dau du dinh sweep cho model DA fine-tune (dong #12), nhung
# fine-tune tren FULL train.json lai cho ket qua THAP HON pretrained (Score PUBLIC 0.8922 vs 0.9058,
# ID 889985) - dong huong thu nghiem fine-tune AITeamVN lai (xem submission-log-task1.md). Vi vay
# script nay sweep cho ban PRETRAINED (cau hinh dang dung that), KHONG phai ban fine-tune.
#
# Dung lai cache co san tu lan chay LOCAL truoc do (outputs/cache/embcompare_aiteamvn.npz) - KHONG
# can encode lai corpus (tiet kiem ~2 tieng so voi encode moi).
#
# Chay tu thu muc GOC UIT-data-science-2026:  .\scripts\sweep_bm25_weight_aiteamvn.ps1

python -m src.evaluation.sweep_bm25_weight --embedding-model AITeamVN/Vietnamese_Embedding --embedding-cache outputs/cache/embcompare_aiteamvn.npz --weights 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2 --query-prefix="" --passage-prefix="" --max-seq-length 2048 --report outputs/predictions/sweep_bm25_weight_aiteamvn_report.json

Write-Host ""
Write-Host "=== Xong. So Recall cao nhat trong bang tren voi moc bm25-weight=0.4 hien dang dung (0.9267 LOCAL) ===" -ForegroundColor Green
Write-Host "Neu dinh o gia tri khac 0.4 va cai thien ro ret (vd +0.01 tro len) - cap nhat lai config dang dung va can nhac nop public xac nhan (neu cai thien lon, vd +0.02-0.03+)." -ForegroundColor Green
