# =============================================================================
# THU RERANKER THEO CACH MOI: la MOT NGUON trong RRF, KHONG thay the xep hang  (2026-08-22)
#
# TAI SAO DANG THU LAI DU DA "DONG" 2 LAN:
#   full_pipeline_eval.py xep lai candidate THUAN theo diem reranker (dong 585:
#   sorted(zip(candidates, scores), key=-score)), VUT BO hoan toan diem RRF.
#   => reranker dang chay o "TRONG SO VO HAN".
#   Day DUNG kieu that bai da chan doan duoc o harrier: unit-test cho thay w=1.5 la nguon phu
#   chiem sach top-5. Va harrier o w=0.8 that bai tren public (-0.0013), o w=0.2 lai THANG (+0.0035).
#   Log du an cung da tu ghi day la ngoai le duy nhat dang thu lai: "ket hop diem thay vi thay the".
#
# KHONG CAN TRAIN GI - model reranker-ft da co san tren dia (fine-tune tu BAAI/bge-reranker-v2-m3,
# hard-negative, tu giai doan 1 cua du an).
#
# Chay tu thu muc GOC:  .\scripts\run_reranker_as_source.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== PHA 1: cham lai top-100 ung vien cua v1 bang reranker-ft (val_split_20) ===" -ForegroundColor Cyan
Write-Host "    (1398 cau x 100 cap = ~140k cap qua cross-encoder, uoc ~5-15 phut tren RTX 3060)" -ForegroundColor DarkGray
# --max-length 256 + fp16 (mac dinh) + top-n 50: nhanh ~8-10x so voi ban dau (fp32/512/100).
# Ha 100->50 KHONG mat gi dang ke: reranker o day chi XEP LAI ung vien cua v1 (khong mo rong pool),
# ma voi rrf_k=60 thi chenh lech giua hang 50 va hang 100 chi la 1/110 so 1/160 - qua nho.
python -m src.evaluation.ensemble_fusion rank-reranker --name rrft `
    --reranker-model models/local_train/reranker-ft `
    --candidates outputs/cache/ranks_v1.npz `
    --top-n 50 --batch-size 64 --max-length 256 `
    --out outputs/cache/ranks_rrft.npz

Write-Host "`n=== PHA 2: do duong cong TRONG SO reranker (khong dung GPU) ===" -ForegroundColor Cyan
# --use-top-n rrft=50 la BAT BUOC: file ranks_rrft.npz chi co 50 cot, trong khi --default-top-n = 100.
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank rrft=outputs/cache/ranks_rrft.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --weights harrier=0.2 `
    --weights rrft=0,0.1,0.2,0.4,0.6,0.8,1.0,1.5 `
    --weights bm25=0.1 `
    --use-top-n rrft=50 `
    --rrf-k 60 `
    --show-top 30 `
    --report outputs/predictions/reranker_source_sweep.json

Write-Host ""
Write-Host "=== CACH DOC ===" -ForegroundColor Green
Write-Host "Dong 'rrft=0' = cau hinh CHINH THUC hien tai (v1 + harrier 0.2 + bm25 0.1)." -ForegroundColor Green
Write-Host "Xem duong cong theo rrft: neu co dinh RO RANG o trong so nho/vua -> reranker CO ich khi" -ForegroundColor Green
Write-Host "chi DONG GOP thay vi thay the => giai thich duoc vi sao 2 lan truoc that bai." -ForegroundColor Green
Write-Host ""
Write-Host "LUU Y: theo quy tac chot 2026-08-21, chenh lech LOCAL <0.01 KHONG du doan duoc public." -ForegroundColor Yellow
Write-Host "  - Neu loi ich >0.02  -> tin duoc, dang nop." -ForegroundColor Yellow
Write-Host "  - Neu loi ich <0.01  -> chi la ung vien, PHAI chon trong so bang suy luan tong quat hoa" -ForegroundColor Yellow
Write-Host "    (giong cach da chon harrier=0.2), dung lay thang argmax cua local." -ForegroundColor Yellow
