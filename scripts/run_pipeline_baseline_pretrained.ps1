# Baseline PRETRAINED (chua fine-tune) - do local tren val_split_20 bang DUNG pipeline hybrid
# (dense + BM25 + RRF + rerank) nhu Giai doan 1/2, nhung dung model GOC (bqbbao6/VN_legal_embedding_512
# + BAAI/bge-reranker-v2-m3), KHONG qua fine-tune. Muc dich: co con so Recall/Precision LOCAL
# "sach" (cung pipeline, cung tap val_split_20, cung prefix) de so truc tiep voi dong #4
# (0.8624/0.1838, model da fine-tune) trong claude/submission-log-task1.md - tach bach ro
# fine-tuning dong gop bao nhieu, thay vi dong #1 cu (do tren tap khac, khong hybrid).
#
# Dong thoi bat --stage-eval de xem Recall/Precision RIENG tung giai doan (dense-only/BM25-only/
# sau hybrid merge/sau rerank+top-k) cua CHINH ban pretrained nay - xem model pretrained yeu o
# giai doan nao truoc khi fine-tune (vd neu rerank pretrained lam giam diem thi biet ngay
# fine-tune reranker la uu tien, khong phai embedding).
#
# KHONG tot lot nop BTC - chi chay LOCAL, khong dung file cau hoi public/khong tao submission.zip.
#
# File nay nam trong thu muc con scripts\ (de gon thu muc goc), nhung PHAI chay tu thu muc GOC
# UIT-data-science-2026 (khong phai tu ben trong scripts\) - vi tat ca duong dan trong script deu
# la duong dan tuong doi (data/processed/..., outputs/...) tinh theo thu muc dang dung (working
# directory), khong phai theo vi tri file .ps1.
#
# Cach chay: mo PowerShell tai thu muc UIT-data-science-2026 (thu muc GOC, KHONG cd vao scripts\),
# go:  .\scripts\run_pipeline_baseline_pretrained.ps1
# Neu bi chan boi execution policy:  powershell -ExecutionPolicy Bypass -File scripts\run_pipeline_baseline_pretrained.ps1

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

# Khong dung --embedding-cache trung voi cac lan chay khac (models/vietnamese-legal-embedding-ft*)
# vi day la embedding KHAC (model pretrained goc, chua fine-tune) - dung nham cache se cho ra
# ket qua cua model fine-tune, khong phai pretrained.
Invoke-Step "1/3 - Chay full pipeline (pretrained, --stage-eval) tren val_preprocessed.jsonl" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model bqbbao6/VN_legal_embedding_512 --reranker-model BAAI/bge-reranker-v2-m3 --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/stage_eval_pretrained.json --embedding-cache outputs/cache_pretrained_stage.npz --stage-eval" `
    "outputs/stage_eval_pretrained.json"

Invoke-Step "2/3 - Bao cao tong hop tung giai doan (dense-only / BM25-only / hybrid / rerank)" `
    "python -m src.evaluation.stage_report --ground-truth data/processed/val_split_20.json --base outputs/stage_eval_pretrained.json --report outputs/stage_eval_pretrained_report.json"

Invoke-Step "3/3 - Cham diem chinh thuc (Recall/Precision cuoi) + phan tich theo n_relevant" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/stage_eval_pretrained.json --report outputs/stage_eval_pretrained_final_report.json"

Write-Host ""
Write-Host "=== HOAN TAT - so ket qua Recall/Precision o tren voi dong #4 (0.8624/0.1838, model da fine-tune) trong submission-log-task1.md ===" -ForegroundColor Green
Write-Host "    Khong can nop BTC - day la chay LOCAL de lay so lieu so sanh, bao lai ket qua de cap nhat bang log." -ForegroundColor DarkGray
