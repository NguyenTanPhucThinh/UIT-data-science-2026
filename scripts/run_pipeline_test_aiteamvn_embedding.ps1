# Test embedding AITeamVN/Vietnamese_Embedding (pretrained, CHUA fine-tune) tren val_split_20,
# dung DUNG cau hinh dang chon lam mac dinh lam viec hien tai (KHONG reranker, bm25-weight=0.4 -
# xem "QUYET DINH HIEN TAI" trong claude/submission-log-task1.md), de so sanh CONG BANG voi:
#   - Model dang dung (fine-tune tren bqbbao6/VN_legal_embedding_512): Recall 0.8919 (dong #10)
#   - Neu muon so ca voi pretrained bqbbao6 goc: chay them script rieng (chua co, hoi Claude neu can)
#
# Quy uoc quan trong cua AITeamVN/Vietnamese_Embedding (da xac nhan qua model card, 2026-08-14):
#   - KHONG can prefix "query: "/"passage: " (doi xung, --query-prefix ""/--passage-prefix "")
#   - Can --max-seq-length 2048 (khuyen nghi cua model card, vi chunk phap luat co the dai)
# NEU sau nay AITeamVN sua doi quy uoc nay (vd ra ban moi), PHAI sua lai script - dung suy doan.
#
# *** SUA LOI (2026-08-14): PowerShell tu dong XOA tham so chuoi rong "" khi goi chuong trinh
# ngoai (python.exe) qua Invoke-Expression - day la loi/hanh vi da biet cua PowerShell, KHONG
# phai loi script. Vi du "--query-prefix ''" (co khoang trang truoc "") se bi PowerShell nuot
# mat tham so, day lech het cac tham so phia sau, gay loi "expected one argument". Cach sua: viet
# DINH LIEN "--query-prefix=''" (KHONG co khoang trang truoc cap nhay kep) - argparse van hieu
# dung la chuoi rong, nhung PowerShell KHONG con nuot mat vi ca cum "--query-prefix=" + "" gop
# thanh 1 token KHONG rong. Neu ban tu sua script nay sau nay va can truyen chuoi rong cho tham
# so khac, LUON dung kieu --ten-tham-so="" (dinh lien, co dau =), KHONG dung --ten-tham-so "".
#
# Bat --stage-eval de xem rieng dense-only/BM25-only/hybrid_fused - vi KHONG dung reranker nen
# "final" (cot cuoi stage_report) se TRUNG voi "hybrid_fused" (dung dung logic da chon hien tai).
#
# KHONG ton luot nop BTC - chi chay LOCAL (val_preprocessed.jsonl + val_split_20.json), khong dung
# file cau hoi public, khong tao submission.zip.
#
# Cach chay: mo PowerShell tai thu muc GOC UIT-data-science-2026, go:
#   .\run_pipeline_test_aiteamvn_embedding.ps1
# Neu bi chan boi execution policy:
#   powershell -ExecutionPolicy Bypass -File run_pipeline_test_aiteamvn_embedding.ps1
#
# LAN CHAY DAU se phai tai model AITeamVN/Vietnamese_Embedding tu HuggingFace (~1-2GB) + encode lai
# toan bo ~351k chunk corpus (khong dung chung cache voi model dang dung - file cache o day la MOI,
# rieng cho model nay) - co the mat vai chuc phut tuy GPU/CPU. Cache duoc luu lai, lan chay sau
# (neu doi tham so khac nhung KHONG doi model) se nhanh hon nhieu.

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

Invoke-Step "1/3 - Chay full pipeline (AITeamVN/Vietnamese_Embedding, KHONG reranker, --stage-eval) tren val_preprocessed.jsonl" `
    "python -m src.evaluation.full_pipeline_eval --embedding-model AITeamVN/Vietnamese_Embedding --chunks data/processed/chunks_corpus.jsonl --questions data/processed/val_preprocessed.jsonl --output outputs/predictions/embcompare_aiteamvn.json --embedding-cache outputs/cache/embcompare_aiteamvn.npz --bm25-weight 0.4 --query-prefix=`"`" --passage-prefix=`"`" --max-seq-length 2048 --stage-eval" `
    "outputs/predictions/embcompare_aiteamvn.json"

Invoke-Step "2/3 - Bao cao tong hop tung giai doan (dense-only / BM25-only / hybrid_fused / final)" `
    "python -m src.evaluation.stage_report --ground-truth data/processed/val_split_20.json --base outputs/predictions/embcompare_aiteamvn.json --report outputs/predictions/embcompare_aiteamvn_stage_report.json"

Invoke-Step "3/3 - Cham diem chinh thuc (Recall/Precision cuoi) + phan tich theo n_relevant" `
    "python -m src.evaluation.evaluate --ground-truth data/processed/val_split_20.json --predictions outputs/predictions/embcompare_aiteamvn.json --report outputs/predictions/embcompare_aiteamvn_final_report.json"

Write-Host ""
Write-Host "=== HOAN TAT - so Recall/Precision o tren voi dong #10 (0.8919/0.1902, model dang dung) trong submission-log-task1.md ===" -ForegroundColor Green
Write-Host "    Neu AITeamVN thang hoac gan bang 0.8919 du CHUA fine-tune -> tin hieu manh de fine-tune tiep." -ForegroundColor DarkGray
Write-Host "    Khong can nop BTC - day la chay LOCAL de lay so lieu so sanh, bao lai ket qua de Claude cap nhat bang log + docs/allowed-models.md." -ForegroundColor DarkGray
