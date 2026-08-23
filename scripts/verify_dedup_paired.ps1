# =============================================================================
# KIEM TRA CAP DOI (paired A/B) cho dedup+backfill tren cau hinh CHINH THUC hien tai
# (v1=1.0, harrier=0.2, bm25=0.1, rrf_k=60) — 2026-08-23
#
# LY DO CAN SCRIPT NAY: lan chay truoc (--dedup-groups, luoi trong so CHI CO 1 gia tri
# moi nguon, KHONG co 0.0 trong luoi) da lam script tinh "baseline" SAI — logic baseline
# trong cmd_fuse chi kiem tra weight=0 O NHUNG NGUON MA LUOI CO CHUA 0.0; vi luoi lan do
# khong nguon nao co 0.0, dieu kien "all(...)" duyet tren TAP RONG -> True VO NGHIA (vacuous
# truth) -> chinh to hop v1=1/harrier=0.2/bm25=0.1 (dedup ON) bi gan nham la "baseline",
# roi lai duoc so sanh voi chinh no -> Delta = 0.0000 va ket luan "DONG huong" la VO GIA TRI,
# khong phai vi dedup that su khong co tac dung.
#
# Script nay tranh bug do bang cach chay 2 LENH RIENG BIET (KHONG dedup, CO dedup) tren
# CUNG 1 cau hinh trong so, roi tu tay tru hieu — khong dua vao auto-detect baseline cua fuse.
#
# Chay tu thu muc GOC:  .\scripts\verify_dedup_paired.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== [A] KHONG dedup (doi chung) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --weights harrier=0.2 `
    --weights bm25=0.1 `
    --rrf-k 60 `
    --report outputs/predictions/dedup_verify_off.json

Write-Host "`n=== [B] CO dedup+backfill (--dedup-groups) ===" -ForegroundColor Cyan
python -m src.evaluation.ensemble_fusion fuse `
    --rank v1=outputs/cache/ranks_v1.npz `
    --rank harrier=outputs/cache/ranks_harrier.npz `
    --rank bm25=outputs/cache/ranks_bm25.npz `
    --weights harrier=0.2 `
    --weights bm25=0.1 `
    --rrf-k 60 `
    --dedup-groups data/processed/duplicate_doc_groups.json `
    --report outputs/predictions/dedup_verify_on.json

Write-Host "`n=== DOC KET QUA VA TU TRU HIEU (KHONG dua vao dong 'baseline' cua fuse o tren) ===" -ForegroundColor Yellow
# LUU Y: PowerShell KHONG co cu phap heredoc kieu bash "<<'EOF'" - ban dau viet nham nen script
# bao loi parse, khong chay duoc. Sua bang cach dung PowerShell here-string (@'...'@) roi PIPE
# thang noi dung do vao "python -" (python doc script tu STDIN khi truyen dau "-" thay ten file).
$pyCode = @'
import json
a = json.load(open("outputs/predictions/dedup_verify_off.json", encoding="utf-8"))
b = json.load(open("outputs/predictions/dedup_verify_on.json", encoding="utf-8"))
ra, rb = a["best"]["avg_recall"], b["best"]["avg_recall"]
print(f"  KHONG dedup : Recall {ra:.4f}")
print(f"  CO dedup    : Recall {rb:.4f}")
print(f"  Delta (dedup - khong dedup) = {rb - ra:+.4f}")
if rb - ra > 0.0005:
    print("  -> dedup+backfill CO GIUP ICH (nho). Can nhac giu.")
elif rb - ra < -0.0005:
    print("  -> dedup+backfill LAM GIAM diem. DONG huong nay, giu nguyen khong dedup.")
else:
    print("  -> KHONG khac biet dang ke (trong nhieu +-0.0005). Coi nhu trung tinh, khong dang de them phuc tap.")
'@
$pyCode | python -
