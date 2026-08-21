# Test embedding AITeamVN/Vietnamese_Embedding_v2 tren val_split_20 (LOCAL, KHONG ton luot nop BTC).
#
# CANH BAO QUAN TRONG (tra model card chinh thuc huggingface.co/AITeamVN/Vietnamese_Embedding_v2,
# 2026-08-17) TRUOC KHI CHAY: v2 KHONG PHAI "v1 nang cap" don gian - v2 fine-tune tu nen BAAI/bge-m3
# (kien truc khac han v1, von dua tren XLM-R-large), train tren tap triplet LON HON v1. Nhung benchmark
# CHINH THUC cua ho tren Zalo Legal 2021 (phap luat) cho thay v1 NHINH HON v2 mot chut:
#   Accuracy@1: v1=0.7274 vs v2=0.7262 | Accuracy@3: v1=0.8992 vs v2=0.8927 | MRR@10: v1=0.8181 vs v2=0.8149
# Model card ho tu ghi: v2 "tot cho domain khac" (generalization rong hon), v1 van manh hon cho phap
# luat cu the. Chenh lech nho, khong ket luan chac, va corpus cuoc thi minh KHAC Zalo Legal 2021 - nen
# VAN DANG THU (khong ton gi vi chi chay LOCAL) nhung HA KY VONG, dung nghi v2 "chac thang".
#
# Quy uoc da GIA DINH (CHUA xac nhan chac chan nhu v1 - can kiem tra neu ket qua bat thuong thap):
#   - max_seq_length 2048 (model card v2 ghi ro, giong v1)
#   - Dim output 1024 (v1 la 1024 cung - khong anh huong code, chi de biet)
#   - KHONG can prefix "query: "/"passage: " - GIA DINH dua tren vi du code chinh thuc cua ho chi
#     goi thang model.encode(sentences) khong prefix (giong quy uoc v1), NHUNG README v2 khong noi
#     ro nhu README v1 (v1 co xac nhan tuong minh, v2 thi KHONG) - can de y neu Recall thap bat
#     thuong so voi dense-only cua v1 (0.9200), co the do gia dinh prefix sai.
#
# Dung bm25-weight=0.1 (cau hinh CHINH THUC hien tai, dong #11b - KHONG dung 0.4 cu da lac hau) va
# --bm25-fast (da verify DUNG + AN TOAN bo nho tren corpus that, xem "--bm25-fast" trong log) de
# chay nhanh hon nhieu so voi ban BM25 cu.
#
# Dung CACHE RIENG (embcompare_aiteamvn_v2.npz) - KHONG dung chung voi cache cua v1
# (embcompare_aiteamvn.npz), vi model khac nhau se ra embedding khac nhau hoan toan.
#
# Cach chay: mo PowerShell tai thu muc GOC UIT-data-science-2026, go:
#   .\scripts\test_aiteamvn_v2_embedding.ps1
# Neu bi chan boi execution policy:
#   powershell -ExecutionPolicy Bypass -File scripts\test_aiteamvn_v2_embedding.ps1
#
# LAN CHAY DAU se tai model AITeamVN/Vietnamese_Embedding_v2 tu HuggingFace (~1-2GB) + encode lai
# TOAN BO ~351k chunk corpus (khong dung chung cache voi v1) - co the mat vai chuc phut. Cache duoc
# luu lai cho lan chay sau (vd neu doi cau hinh khac nhung khong doi model).

function Invoke-Step($desc, $cmd, $skipIfExists = $null) {
    Write-Host ""
    if ($skipIfExists -and (Test-Path $skipIfExists) -and (Get-ChildItem $skipIfExists -ErrorAction SilentlyContinue)) {
        Write-Host "=== $desc (BO QUA - da co san tai $skipIfExists) ===" -ForegroundColor Yellow
        return
    }
    Write-Host "=== $desc ===" -ForegroundColor Cyan
    Write-Host $cmd -ForegroundColor DarkGray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "LOI o buoc: $desc (exit code $LASTEXITCODE) - dung lai, khong chay tiep buoc sau." -ForegroundColor Red
        exit 1
    }
}

Invoke-Step "1/3 - Chay full pipeline (AITeamVN/Vietnamese_Embedding_v2, KHONG reranker, bm25-weight=0.1, --bm25-fast, --stage-eval) tren val_preprocessed.jsonl" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model AITeamVN/Vietnamese_Embedding_v2 --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/embcompare_aiteamvn_v2.json --embedding-cache outputs/cache/embcompare_aiteamvn_v2.npz --bm25-weight 0.1 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048 --stage-eval --bm25-fast" `
    "outputs/predictions/embcompare_aiteamvn_v2.json"

Invoke-Step "2/3 - Bao cao tong hop tung giai doan (dense-only / BM25-only / hybrid_fused / final)" `
    "python -m src.evaluation.stage_report --ground-truth data/processed/val_split_20.json --base outputs/predictions/embcompare_aiteamvn_v2.json --report outputs/predictions/embcompare_aiteamvn_v2_stage_report.json"

Invoke-Step "3/3 - Cham diem chinh thuc (Recall/Precision cuoi) + phan tich theo n_relevant" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/embcompare_aiteamvn_v2.json --report outputs/predictions/embcompare_aiteamvn_v2_final_report.json"

Write-Host ""
Write-Host "=== HOAN TAT - so Recall/Precision o tren voi dong #11b (v1 pretrained, bm25=0.1): Recall 0.9303 / Precision 0.1983 ===" -ForegroundColor Green
Write-Host "    Cung nen so rieng Dense-only (giai doan 1) voi dense-only cua v1: 0.9200 - de tach bach" -ForegroundColor DarkGray
Write-Host "    xem v2 tot/kem hon o CHINH embedding, hay chi khac o cach BM25/hybrid tuong tac." -ForegroundColor DarkGray
Write-Host ""
Write-Host "    KHONG tu chay submit_aiteamvn_v2_public.ps1 ngay - xem so lieu LOCAL nay truoc, bao lai" -ForegroundColor Yellow
Write-Host "    cho Claude (hoac tu quyet dinh) co dang ton 1 luot nop public de xac nhan hay khong," -ForegroundColor Yellow
Write-Host "    dung ky luat cua du an (luon do LOCAL truoc khi ton luot nop)." -ForegroundColor Yellow
