# =============================================================================
# NHANH BGE-M3 SPARSE (lexical hoc duoc) LAM NGUON THU 4 CHO ENSEMBLE — 2026-08-27
#
# VI SAO HUONG NAY: pipeline dang co v1 (dense) + harrier (dense) + BM25 (lexical THONG KE).
#   bge-m3 sparse la lexical HOC DUOC -> LOAI TIN HIEU MOI, khong phai bien the cua cai da co.
#   Lich su du an: MOI lan tang diem deu den tu doi TIN HIEU vao fusion hoac CACH tron chung
#   (+0.0586 doi embedding, +0.0115 sweep bm25-weight, +0.0035 them nhanh harrier). CHUA lan nao
#   den tu them tang xu ly (reranker 4/4 that bai) hay train lai (fine-tune 4/4 that bai).
#
# TUAN THU: BAAI/bge-m3 ~568M -> tong ensemble ~1,73 ty, VAN DUOI han muc 4 ty (muc 5).
#   NEU chot dung cho vong private: PHAI khai bao trong scripts/allowed-models.md + de xuat BTC
#   truoc 10 ngay han chot (muc 7). Deadline public: 18/09/2026 23:59 GMT+7.
#
# KHONG CAN CAI THEM GOI NAO. Script python tu tai file sparse_linear.pt (3,52 kB) tu HuggingFace
#   va tu ap dung -> chi dung torch + transformers DA CO trong .venv. CO Y khong dung thu vien
#   FlagEmbedding vi no keo theo rang buoc phien ban transformers/peft, rui ro lam hong moi truong
#   dang chay tot cho toan bo pipeline.
#
# 3 BUG DA BAT DUOC BANG TEST TRUOC KHI GIAO (xem src/evaluation/bgem3_sparse.py):
#   1. Thieu .detach() -> tensor mang do thi gradient -> .numpy() nem RuntimeError.
#   2. top_n=200 nhung corpus thu chi co 60 chunk -> "could not broadcast" khi gan mang.
#   3. Thieu sap xep theo do dai -> padding lang phi, encode cham nhieu lan. Da them, VA da test
#      rieng rang thu tu dong duoc khoi phuc dung 100% (sai buoc nay la lech het chi so chunk).
#
# CACH DUNG (chay tu thu muc GOC):
#   .\scripts\run_bgem3_sparse.ps1                  <- mac dinh: chi chay KIEM TRA (vai phut)
#   .\scripts\run_bgem3_sparse.ps1 -Stage encode    <- job dai ~1,5-2h GPU, CO THE CHAY TIEP
#   .\scripts\run_bgem3_sparse.ps1 -Stage rank
#   .\scripts\run_bgem3_sparse.ps1 -Stage fuse
# =============================================================================
param(
    [ValidateSet("check", "encode", "rank", "fuse")]
    [string]$Stage = "check",
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$SHARDS = "outputs/cache/bgem3_sparse_shards"
$SMOKE = "outputs/cache/bgem3_sparse_smoke"

# $ErrorActionPreference="Stop" KHONG bat ma loi cua chuong trinh ngoai (python) - phai tu kiem
# $LASTEXITCODE. Thieu cai nay khien lan chay 2026-08-28 in ra "KIEM TRA XONG" du buoc rank DA LOI.
function Assert-Ok($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[DUNG] Buoc '$what' that bai (exit $LASTEXITCODE). Doc loi o tren, DUNG chay tiep." -ForegroundColor Red
        exit 1
    }
}
# Ep Python xuat UTF-8 de tieng Viet khong bi vo khi in ra console Windows.
$env:PYTHONIOENCODING = "utf-8"

if ($Stage -eq "check") {
    Write-Host "`n=== [1/3] SELFTEST: kiem tra logic sparse (khong GPU, khong tai model) ===" -ForegroundColor Cyan
    python -m src.evaluation.bgem3_sparse selftest
    Assert-Ok "selftest"

    Write-Host "`n=== [2/3] INSPECT: mat thuong kiem tra sparse head (lan dau se tai model ~2,3GB) ===" -ForegroundColor Cyan
    Write-Host "    (dung --from-questions doc TRUC TIEP tu file JSONL - TRANH viec go tieng Viet" -ForegroundColor DarkGray
    Write-Host "     co dau qua dong lenh PowerShell, thu de bi HONG MA truoc khi toi model)" -ForegroundColor DarkGray
    python -m src.evaluation.bgem3_sparse inspect --fp16 --top 20 --from-questions 0
    Assert-Ok "inspect"

    Write-Host "`n>>> DUNG LAI DOC KET QUA TREN <<<" -ForegroundColor Yellow
    Write-Host "Dong dau tien (cau hoi) PHAI hien tieng Viet co dau BINH THUONG - neu no hien" -ForegroundColor Yellow
    Write-Host "thanh ky tu la (Ã, Â, á»...) thi BAO LAI NGAY, dung doc tiep danh sach token." -ForegroundColor Red
    Write-Host "DUNG neu: token trong so cao nhat la tu NOI DUNG co nghia (danh tu, thuat ngu)." -ForegroundColor Yellow
    Write-Host "SAI neu : toan dau cau / <s> </s> <pad>, hoac moi trong so xap xi bang nhau -> BAO LAI, DUNG chay tiep." -ForegroundColor Red

    Write-Host "`n=== [3/3] SMOKE TEST tren du lieu THAT (2000 chunk, 50 cau hoi, ~2-3 phut) ===" -ForegroundColor Cyan
    Write-Host "    (ghi vao thu muc shard RIENG '$SMOKE' de KHONG lam ban shard cua lan chay that)" -ForegroundColor DarkGray
    python -m src.evaluation.bgem3_sparse encode --fp16 --batch-size $BatchSize `
        --shard-dir $SMOKE --limit-chunks 2000 --shard-size 1000
    Assert-Ok "smoke encode"
    # --limit-chunks PHAI khop dung voi lenh encode o tren, neu khong chot chan lech so dong se
    # dung chuong trinh (dung thiet ke) - day CHINH LA cho lan chay 2026-08-28 dung lai giua chung.
    python -m src.evaluation.bgem3_sparse rank --fp16 --batch-size $BatchSize `
        --shard-dir $SMOKE --limit-chunks 2000 --limit-questions 50 --top-n 50 `
        --chunks data/processed/chunks_corpus.jsonl `
        --out outputs/cache/ranks_bgem3sparse_smoke.npz
    Assert-Ok "smoke rank"

    Write-Host "`n=== KIEM TRA XONG - CA 3 BUOC DEU DAT ===" -ForegroundColor Green
    Write-Host "Luu y: smoke chi co 2000/351394 chunk nen Recall se RAT THAP - dieu do BINH THUONG." -ForegroundColor DarkGray
    Write-Host "Chi can no CHAY DUOC va Recall > 0 la du de sang buoc that." -ForegroundColor DarkGray
    Write-Host "`nBuoc tiep (job dai ~1,5-2h GPU, CO THE ngat giua chung roi chay lai, khong mat gi):" -ForegroundColor Green
    Write-Host "  .\scripts\run_bgem3_sparse.ps1 -Stage encode" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "encode") {
    Write-Host "`n=== ENCODE TOAN CORPUS 351.394 chunk (~1,5-2h GPU) ===" -ForegroundColor Cyan
    Write-Host "AN TOAN: chia shard 20.000 chunk, shard nao xong luu ngay ra dia. Neu bi ngat" -ForegroundColor Green
    Write-Host "(Ctrl+C, mat dien, OOM), chay LAI DUNG lenh nay -> tu bo qua shard da co." -ForegroundColor Green
    Write-Host "Neu OOM VRAM: chay lai voi -BatchSize 4 (hoac 2). Shard da xong van duoc giu." -ForegroundColor Yellow
    python -m src.evaluation.bgem3_sparse encode --fp16 --batch-size $BatchSize --shard-dir $SHARDS
    Assert-Ok "encode"
    Write-Host "`nTiep theo:  .\scripts\run_bgem3_sparse.ps1 -Stage rank" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "rank") {
    Write-Host "`n=== XEP HANG tren val_split_20 (1398 cau hoi) ===" -ForegroundColor Cyan
    python -m src.evaluation.bgem3_sparse rank --fp16 --batch-size $BatchSize `
        --shard-dir $SHARDS --top-n 200 --out outputs/cache/ranks_bgem3sparse.npz
    Assert-Ok "rank"
    Write-Host "`nDoc dong 'Recall@5 DUNG RIENG':" -ForegroundColor Yellow
    Write-Host "  ~0.00        -> sparse head HONG, BAO LAI, dung fuse." -ForegroundColor Red
    Write-Host "  0.5 - 0.9    -> BINH THUONG, dung ky vong bang v1 (0.9267). Sang buoc fuse." -ForegroundColor Green
    Write-Host "`nTiep theo:  .\scripts\run_bgem3_sparse.ps1 -Stage fuse" -ForegroundColor Green
    exit 0
}

if ($Stage -eq "fuse") {
    Write-Host "`n=== FUSE: giu nguyen harrier=0.2 / bm25=0.1 / rrf_k=60, SWEEP rieng trong so sparse ===" -ForegroundColor Cyan
    python -m src.evaluation.ensemble_fusion fuse `
        --rank v1=outputs/cache/ranks_v1.npz `
        --rank harrier=outputs/cache/ranks_harrier.npz `
        --rank bm25=outputs/cache/ranks_bm25.npz `
        --rank m3sparse=outputs/cache/ranks_bgem3sparse.npz `
        --weights harrier=0.2 `
        --weights bm25=0.1 `
        --weights m3sparse=0,0.05,0.1,0.2,0.4,0.6,0.8,1.0 `
        --rrf-k 60 `
        --report outputs/predictions/ensemble_add_m3sparse_report.json
    Assert-Ok "fuse"

    Write-Host "`n=== CACH DOC KET QUA ===" -ForegroundColor Yellow
    Write-Host "1. Dong 'm3sparse=0' PHAI ra DUNG 0.9319 (= cau hinh chinh thuc hien tai)." -ForegroundColor Green
    Write-Host "   Neu KHONG dung 0.9319 -> co loi o dau do, BAO LAI, DUNG tin bang ket qua." -ForegroundColor Red
    Write-Host "   (Luoi CO chua 0 nen bay duoc bug 'baseline gia' cua cmd_fuse - xem submission_log.)" -ForegroundColor DarkGray
    Write-Host "2. Quy tac tham dinh da chot: chenh lech LOCAL < 0.01 KHONG du tin cay de doan PUBLIC." -ForegroundColor Yellow
    Write-Host "   >= +0.02  -> tin duoc, dang nop public." -ForegroundColor Green
    Write-Host "   +0.01..0.02 -> vung xam, can can nhac (bm25-weight/ensemble-weight anh huong DEU" -ForegroundColor Yellow
    Write-Host "                  len moi cau hoi nen dang tin hon entity-boost kieu vai case le te)." -ForegroundColor Yellow
    Write-Host "   < +0.01   -> CHUA du bang chung, KHONG ket luan 'cai thien', dung dot luot nop." -ForegroundColor Red
    Write-Host "3. Neu giam DON DIEU tu trong so nho nhat -> DONG huong (giong het gte-Qwen2)." -ForegroundColor Yellow
    exit 0
}
