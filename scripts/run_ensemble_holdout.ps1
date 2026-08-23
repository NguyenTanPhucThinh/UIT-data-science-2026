# =============================================================================
# CHAN DOAN: ensemble that su co tot khong, hay chi la OVERFIT val_split_20?  (2026-08-21)
#
# BOI CANH: ensemble (harrier=0.8, rrf_k=8) cho LOCAL 0.9360 tren val_split_20 (+0.0065, p=0.033)
# nhung PUBLIC chi 0.9160 (THAP HON baseline #11b 0.9173). Can biet nguyen nhan la:
#   (A) OVERFIT: ta sweep 96 to hop roi lay max tren val_split_20 (1398 cau) -> so dep la may man.
#   (B) LECH PHAN BO: cai thien la THAT tren du lieu kieu train.json, nhung public-official.json
#       co phan bo khac han nen khong ap dung duoc.
#
# CACH PHAN BIET: do lai tren 5602 cau hoi CON LAI cua train.json (train_split_80) - tap nay:
#   - HOAN TOAN khong giao voi val_split_20 (da kiem tra: giao = 0)
#   - Lon gap 4 lan val_split_20 -> nhieu it hon han
#   - KHONG ro ri: moi model dang dung deu PRETRAINED, khong fine-tune tren bat ky phan nao cua train.json
#
# DOC KET QUA:
#   - Neu (harrier=0.8, rrf_k=8) cung cho ~+0.006 o day  -> KHONG overfit -> nguyen nhan la (B) LECH PHAN BO
#     => bai hoc lon: val_split_20 (va ca train.json noi chung) KHONG phai proxy dang tin cho public.
#   - Neu KHONG tai lap duoc (~0 hoac am)                 -> (A) OVERFIT -> dong han huong ensemble,
#     va tu nay MOI cai thien nho tren val phai kiem chung tren holdout truoc khi ton luot nop.
#
# Chay tu thu muc GOC:  .\scripts\run_ensemble_holdout.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$Q  = "data/processed/train_split_80_questions.jsonl"
$GT = "data/processed/train_split_80.json"

if (-not (Test-Path $Q)) { Write-Host "[loi] Thieu $Q - bao lai cho Claude de tao lai." -ForegroundColor Red; exit 1 }

Write-Host "`n=== PHA 1a: thu hang v1 tren 5602 cau holdout ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name v1 `
    --embedding-model AITeamVN/Vietnamese_Embedding `
    --embedding-cache outputs/cache/embcompare_aiteamvn.npz `
    --questions $Q `
    --query-prefix="" --passage-prefix="" `
    --max-seq-length 2048 `
    --out outputs/cache/ranks_hold_v1.npz

Write-Host "`n=== PHA 1b: thu hang harrier tren 5602 cau holdout ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name harrier `
    --embedding-model mainguyen9/vietlegal-harrier-0.6b --trust-remote-code `
    --embedding-cache outputs/cache/embcompare_vietlegal_harrier.npz `
    --questions $Q `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" `
    --max-seq-length 1024 `
    --out outputs/cache/ranks_hold_harrier.npz

Write-Host "`n=== PHA 1c: thu hang BM25 tren 5602 cau holdout ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank-bm25 --name bm25 `
    --questions $Q `
    --out outputs/cache/ranks_hold_bm25.npz

Write-Host "`n=== PHA 2: do lai DUNG cac to hop then chot tren holdout ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_hold_v1.npz `
    --rank harrier=outputs/cache/ranks_hold_harrier.npz `
    --rank bm25=outputs/cache/ranks_hold_bm25.npz `
    --ground-truth $GT `
    --weights harrier=0,0.2,0.4,0.6,0.8,1.0 `
    --weights bm25=0.1 `
    --rrf-k 60,30,10,8,5 `
    --show-top 30 `
    --report outputs/predictions/ensemble_holdout_report.json

Write-Host ""
Write-Host "=== CACH DOC ===" -ForegroundColor Green
Write-Host "1. Dong 'harrier=0 rrf_k=60' = baseline #11b tren tap NAY (chua tung do, se KHAC 0.9295)." -ForegroundColor Green
Write-Host "2. So dong 'harrier=0.8 rrf_k=8' voi baseline do:" -ForegroundColor Green
Write-Host "   - Neu cung hon ~+0.006  -> KHONG overfit, nguyen nhan la LECH PHAN BO train vs public." -ForegroundColor Green
Write-Host "   - Neu ~0 hoac am        -> OVERFIT val_split_20 -> dong han huong ensemble." -ForegroundColor Yellow
Write-Host "3. Da them w=0.2/0.4 (chua tung thu): neu trong so NHO van giu duoc loi ich thi con cua" -ForegroundColor Green
Write-Host "   - vi harrier tut diem tren public manh gap doi v1 (-0.0265 vs -0.0130)." -ForegroundColor Green
