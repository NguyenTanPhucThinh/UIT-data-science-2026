# =============================================================================
# THU NGHIEM: them gte-Qwen2-1.5B-instruct lam NHANH THU 4 vao ensemble (2026-08-23)
#
# BOI CANH: dong #13 (2026-08-16) da test gte-Qwen2-1.5B-instruct (~1,543 ty tham so, GAP 3 LAN
# v1) DUNG RIENG - Recall 0.8864, THAP HON v1, nhung bi CONFOUND boi truncation (--max-seq-length
# 512, 21,63% chunk bi cat cut). Luc do KET LUAN "KHONG THE KET LUAN CHAC CHAN", TAM dong lai vi do
# cong bang hon can ~21h GPU. NHUNG: chua bao gio thu no o VAI TRO ENSEMBLE (chi thu DUNG RIENG).
# Phan tich complementarity da co san tu 2026-08-21 (xem docstring ensemble_fusion.py) cho thay no
# cuu duoc 25,4% tai lieu ma v1 bo sot - duoi nguong 30-40% nhung KHONG PHAI 0 (gan bang bge-m3's
# 26,8%, model DA bi loai vi cung ho v1). gte-Qwen2 kien truc HOAN TOAN KHAC (Qwen decoder/last-token
# pooling vs XLM-R encoder cua v1/harrier) - dang thu theo dung logic da giup harrier thanh cong.
#
# CHI PHI: RE - dung lai cache corpus DA CO SAN (outputs/cache/gte_qwen2_corpus.npz, ~1,28GB, encode
# tu lan test dong #13) - KHONG can encode lai 351K chunk. Chi can encode 1398 cau hoi (vai phut).
# Neu ket qua co tin hieu duong, MOI dang encode lai corpus o max-seq-length cao hon (~21h GPU) de
# co phep do CONG BANG that su - buoc do KHONG chay trong script nay.
#
# TUAN THU: v1 (568M) + harrier (596M) + gte-Qwen2 (1543M) = ~2,71 ty, VAN DUOI han muc 4 ty.
#
# Chay tu thu muc GOC:  .\scripts\run_ensemble_add_gteqwen2.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== PHA 1: thu hang gte-Qwen2 tren val_split_20 (dung lai cache corpus co san) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion rank --name gteqwen2 `
    --embedding-model Alibaba-NLP/gte-Qwen2-1.5B-instruct --trust-remote-code `
    --embedding-cache outputs/cache/gte_qwen2_corpus.npz `
    --query-instruct "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question" `
    --passage-prefix="" --max-seq-length 512 `
    --out outputs/cache/ranks_gteqwen2.npz

Write-Host "`n=== PHA 2: fuse - GIU NGUYEN harrier=0.2/bm25=0.1/rrf_k=60 (da xac nhan public), sweep RIENG trong so gteqwen2 ===" -ForegroundColor Cyan
# Luu y "cua so trong so hep" da phat hien (xem docstring ensemble_fusion.py): trong so can ~0.6-1.2
# moi du de "mo rong pool" (keo tai lieu v1 bo sot hoan toan vao) - duoi 0.5 thuong VO TAC DUNG.
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --rank gteqwen2=outputs/cache/ranks_gteqwen2.npz `
    --weights harrier=0.2 `
    --weights bm25=0.1 `
    --weights gteqwen2=0,0.2,0.4,0.6,0.8,1.0,1.2 `
    --rrf-k 60 `
    --report outputs/predictions/ensemble_add_gteqwen2_report.json

Write-Host ""
Write-Host "=== DOC KET QUA ===" -ForegroundColor Yellow
Write-Host "Dong 'gteqwen2=0' PHAI ra dung 0.9319 (= cau hinh chinh thuc hien tai, dong #21 do tren val)." -ForegroundColor Green
Write-Host "Neu KHONG ra dung 0.9319 -> co gi do sai, BAO LAI cho Claude, dung tin bang ket qua." -ForegroundColor Yellow
Write-Host ""
Write-Host "Neu co trong so gteqwen2 > 0 vuot 0.9319 ro ret (>= +0.005): DANG can nhac encode lai corpus" -ForegroundColor Green
Write-Host "o max-seq-length cao hon (~1024) de co phep do KHONG bi confound truncation, roi sweep lai." -ForegroundColor Green
Write-Host "Neu KHONG trong so nao vuot: DONG huong nay (gte-Qwen2 khong dang encode lai corpus)." -ForegroundColor Green
