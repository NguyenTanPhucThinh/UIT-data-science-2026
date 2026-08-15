param(
    [string]$Questions = "data/processed/public_test_2_preprocessed.jsonl",

    [string]$QuestionReferenceJson = "data/raw/public-official-2.json",
    [string]$Output = "outputs/task2/submission.json",
    [string]$Chunks = "data/processed/chunks_corpus.jsonl",
    [string]$EmbeddingModel = "models/vietnamese-legal-embedding-ft-full",
    [string]$RerankerModel = "models/reranker-ft-full",
    [string]$LlmModel = "AITeamVN/Vi-Qwen2-1.5B-RAG",
    [string]$EmbeddingCache = "outputs/cache/chunks_corpus_embeddings__ft_full.npz",
    [switch]$LocalFilesOnly,
    [switch]$Package
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath "requirements.txt")) {
    throw "Run this script from the repository root."
}

if (-not (Test-Path -LiteralPath $Questions)) {
    throw "Task 2 questions not found: $Questions. Run build_pipeline.py --task 2 first."
}
if (-not (Test-Path -LiteralPath $Chunks)) {
    throw "Shared chunks corpus not found: $Chunks. Run build_pipeline.py --task 1, 2, or all first."
}
if ($EmbeddingModel -match '^models[\\/]' -and -not (Test-Path -LiteralPath $EmbeddingModel)) {
    throw "Retrieval embedding checkpoint not found: $EmbeddingModel. Complete section 6 (copy or fine-tune the checkpoint) before running Task 2."
}
if ($RerankerModel -match '^models[\\/]' -and -not (Test-Path -LiteralPath $RerankerModel)) {
    throw "Retrieval reranker checkpoint not found: $RerankerModel. Complete section 6 (copy or fine-tune the checkpoint) before running Task 2."
}

$outputParent = Split-Path -Parent $Output
if ($outputParent) {
    New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
}

$retrievalContext = "outputs/task2/retrieval_context.json"
$generationCheckpoint = "outputs/task2/generation_checkpoint.jsonl"

$runArgs = @(
    "-m", "src.evaluation.run_legalqa",
    "--questions", $Questions,
    "--chunks", $Chunks,
    "--embedding-model", $EmbeddingModel,
    "--reranker-model", $RerankerModel,
    "--llm-model", $LlmModel,
    "--embedding-cache", $EmbeddingCache,
    "--output", $Output,
    "--retrieval-context-output", $retrievalContext,
    "--generation-checkpoint", $generationCheckpoint,
    "--diagnostics-output", "outputs/task2/diagnostics.json",
    "--top-k", "5",
    "--candidate-docs", "20",
    "--max-documents", "5",
    "--max-chunks-per-document", "3",
    "--neighbor-window", "1",
    "--max-context-tokens", "4096",
    "--max-new-tokens", "384",
    "--answer-batch-size", "1"
)
if (Test-Path -LiteralPath $retrievalContext) {
    Write-Host "Reusing completed retrieval context: $retrievalContext" -ForegroundColor Green
    $runArgs += "--reuse-retrieval-context"
}
if (Test-Path -LiteralPath $generationCheckpoint) {
    Write-Host "Resuming generated answers from: $generationCheckpoint" -ForegroundColor Green
    $runArgs += "--resume-generation"
}
if ($LocalFilesOnly) {
    $runArgs += "--local-files-only"
}

Write-Host "Running local Task 2 LegalQA pipeline..." -ForegroundColor Cyan
& python @runArgs
if ($LASTEXITCODE -ne 0) {
    throw "Task 2 pipeline failed with exit code $LASTEXITCODE"
}

$outputDirectory = Split-Path -Parent $Output
if (-not $outputDirectory) {
    $outputDirectory = "."
}
$outputBaseName = [System.IO.Path]::GetFileNameWithoutExtension($Output)
$outputExtension = [System.IO.Path]::GetExtension($Output)
$rawOutput = Join-Path $outputDirectory ($outputBaseName + "_raw" + $outputExtension)
$postprocessReport = Join-Path $outputDirectory "postprocess_report.json"
Copy-Item -LiteralPath $Output -Destination $rawOutput -Force

Write-Host "Removing metric-unfriendly LegalQA preambles..." -ForegroundColor Cyan
& python -m src.evaluation.postprocess_legalqa_submission `
    --input $rawOutput `
    --output $Output `
    --report $postprocessReport
if ($LASTEXITCODE -ne 0) {
    throw "Task 2 postprocessing failed with exit code $LASTEXITCODE. Raw output is preserved at $rawOutput"
}

$validateArgs = @(
    "-m", "src.evaluation.validate_legalqa_submission",
    "--predictions", $Output
)
if ($QuestionReferenceJson) {
    $validateArgs += @("--questions", $QuestionReferenceJson)
}

Write-Host "Validating Task 2 submission..." -ForegroundColor Cyan
& python @validateArgs
if ($LASTEXITCODE -ne 0) {
    throw "Task 2 validation failed with exit code $LASTEXITCODE"
}

if ($Package) {
    $packageDir = "outputs/task2/package"
    New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
    Copy-Item -LiteralPath $Output -Destination "$packageDir/submission.json" -Force
    Compress-Archive -Path "$packageDir/submission.json" -DestinationPath "outputs/task2/submission.zip" -Force
    Write-Host "Created outputs/task2/submission.zip" -ForegroundColor Green
}

Write-Host "Task 2 pipeline completed: $Output" -ForegroundColor Green
