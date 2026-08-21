# Test embedding BAAI/bge-m3 (GOC, multilingual, CHUA ai fine-tune rieng cho tieng Viet/phap luat)
# tren val_split_20 (LOCAL, KHONG ton luot nop BTC).
#
# QUAN TRONG - phan biet voi 2 model BAAI khac DA test truoc do (dung nham la "da thu roi"):
#   - BAAI/bge-reranker-v2-m3 (RERANKER, da test dong #8/#14, thua, da dong huong reranker)
#   - AITeamVN/Vietnamese_Embedding_v2 (EMBEDDING, fine-tune TU NEN bge-m3, da test dong #18, thua v1)
#   - BAAI/bge-m3 (EMBEDDING GOC, CHUA ai fine-tune them cho tieng Viet/phap luat) - CHUA TUNG test,
#     day la lan dau tien.
#
# Quy uoc da xac nhan qua model card chinh thuc (huggingface.co/BAAI/bge-m3, 2026-08-17):
#   - Kien truc: XLM-RoBERTa (cung ho tokenizer voi AITeamVN v1, vi AITeamVN v1 cung dua tren XLM-R-large)
#   - max_seq_length ho tro toi 8192, nhung dung 2048 (giong AITeamVN v1) de so sanh CONG BANG cung
#     mot muc cat cut, khong phai vi 2048 la gioi han cua model nay
#   - KHONG can prefix "query: "/"passage: " - model card ghi ro: "no longer requires adding
#     instructions to the queries" (khac ban bge cu hon nhu bge-large can prefix "Represent this
#     sentence for searching relevant passages: ")
#   - Dim dense output: 1024 (giong AITeamVN v1 VA v2, khong anh huong code)
#   - Script nay CHI dung nhanh DENSE cua bge-m3 (qua sentence-transformers .encode() chuan, giong
#     cach full_pipeline_eval.py dang dung cho moi model khac) - KHONG dung nhanh sparse/ColBERT
#     rieng cua bge-m3 (can thu vien FlagEmbedding, kien truc khac han pipeline hybrid dense+BM25
#     dang co san - de nhat quan voi cach so sanh cac model embedding khac trong du an).
#
# Dung bm25-weight=0.1 (cau hinh CHINH THUC hien tai, dong #11b) va --bm25-fast.
# Dung CACHE RIENG (embcompare_bge_m3.npz) - KHONG dung chung voi cache cua AITeamVN v1/v2.
#
# Cach chay: mo PowerShell tai thu muc GOC UIT-data-science-2026, go:
#   .\scripts\test_bge_m3_embedding.ps1
# Neu bi chan boi execution policy:
#   powershell -ExecutionPolicy Bypass -File scripts\test_bge_m3_embedding.ps1
#
# LAN CHAY DAU se tai model BAAI/bge-m3 tu HuggingFace (~2.2GB) + encode lai TOAN BO ~351k chunk
# corpus (cache rieng, khong dung chung voi model khac) - co the mat vai chuc phut. Cache duoc luu
# lai cho lan chay sau.

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

Invoke-Step "1/3 - Chay full pipeline (BAAI/bge-m3, KHONG reranker, bm25-weight=0.1, --bm25-fast, --stage-eval) tren val_preprocessed.jsonl" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model BAAI/bge-m3 --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/embcompare_bge_m3.json --embedding-cache outputs/cache/embcompare_bge_m3.npz --bm25-weight 0.1 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048 --stage-eval --bm25-fast" `
    "outputs/predictions/embcompare_bge_m3.json"

Invoke-Step "2/3 - Bao cao tong hop tung giai doan (dense-only / BM25-only / hybrid_fused / final)" `
    "python -m src.evaluation.stage_report --ground-truth data/processed/val_split_20.json --base outputs/predictions/embcompare_bge_m3.json --report outputs/predictions/embcompare_bge_m3_stage_report.json"

Invoke-Step "3/3 - Cham diem chinh thuc (Recall/Precision cuoi) + phan tich theo n_relevant" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/embcompare_bge_m3.json --report outputs/predictions/embcompare_bge_m3_final_report.json"

Write-Host ""
Write-Host "=== HOAN TAT - so Recall/Precision o tren voi dong #11b (v1 AITeamVN pretrained, bm25=0.1): Recall 0.9303 / Precision 0.1980 ===" -ForegroundColor Green
Write-Host "    Cung nen so rieng Dense-only (giai doan 1) voi dense-only cua v1 (0.9200) va v2 (0.9006)" -ForegroundColor DarkGray
Write-Host "    de xem bge-m3 GOC (chua fine-tune tieng Viet) manh/yeu the nao so voi ca 2 ban da tune." -ForegroundColor DarkGray
Write-Host ""
Write-Host "    KHONG tu chay submit_bge_m3_public.ps1 ngay - xem so lieu LOCAL nay truoc, bao lai cho" -ForegroundColor Yellow
Write-Host "    Claude (hoac tu quyet dinh) co dang ton 1 luot nop public de xac nhan hay khong, dung" -ForegroundColor Yellow
Write-Host "    ky luat cua du an (luon do LOCAL truoc khi ton luot nop)." -ForegroundColor Yellow
