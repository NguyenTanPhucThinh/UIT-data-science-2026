param(
    [string]$Questions = "data/processed/public_test_2_preprocessed.jsonl",
    [string]$QuestionReferenceJson = "data/raw/public-official-2.json",
    [string]$Chunks = "data/processed/chunks_corpus.jsonl",
    [string]$Output = "outputs/task2/submission.json",
    [string]$AnchorModel = "AITeamVN/Vietnamese_Embedding",
    [string]$HarrierModel = "mainguyen9/vietlegal-harrier-0.6b",
    [string]$LlmModel = "AITeamVN/Vi-Qwen2-1.5B-RAG",
    [ValidateSet("baseline", "answer_first", "legal_structured", "grounded_concise", "full")]
    [string]$PromptVariant = "baseline",
    [string]$AnchorEmbeddingCache = "outputs/cache/embcompare_aiteamvn.npz",
    [string]$HarrierEmbeddingCache = "outputs/cache/embcompare_vietlegal_harrier.npz",
    [string]$AnchorRanks = "outputs/cache/ranks_task2_aiteamvn.npz",
    [string]$HarrierRanks = "outputs/cache/ranks_task2_harrier.npz",
    [string]$Bm25Ranks = "outputs/cache/ranks_task2_bm25.npz",
    [string]$RetrievalContext = "outputs/task2/retrieval_context_ensemble.json",
    [string]$GenerationCheckpoint = "outputs/task2/generation_checkpoint_ensemble.jsonl",
    [int]$EmbeddingBatchSize = 64,
    [int]$QuestionBatchSize = 64,
    [int]$AnswerBatchSize = 1,
    [string]$PythonExecutable = "",
    [switch]$RebuildRetrieval,
    [switch]$RebuildEmbeddingCaches,
    [switch]$RebuildGeneration,
    [switch]$LocalFilesOnly,
    [switch]$Package
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExecutable {
    param([string]$RequestedPath)
    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath)) {
            throw "-PythonExecutable does not exist: $RequestedPath"
        }
        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }
    $repoVenvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $repoVenvPython) {
        return (Resolve-Path -LiteralPath $repoVenvPython).Path
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Python was not found. Activate .venv or pass -PythonExecutable <path-to-python.exe>."
    }
    return $command.Source
}

$PythonExecutable = Resolve-PythonExecutable $PythonExecutable
Write-Host "Using Python: $PythonExecutable" -ForegroundColor DarkGray

function Invoke-PythonStep {
    param([string]$Title, [string[]]$Arguments)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath "requirements.txt")) {
    throw "Run this script from the repository root."
}
if (-not (Test-Path -LiteralPath $Questions)) {
    throw "Task 2 questions not found: $Questions. Run build_pipeline.py --task 2 first."
}
if (-not (Test-Path -LiteralPath $Chunks)) {
    throw "Shared chunks corpus not found: $Chunks. Run build_pipeline.py --task all first."
}

$pathsToCreate = @(
    $Output, $AnchorEmbeddingCache, $HarrierEmbeddingCache,
    $AnchorRanks, $HarrierRanks, $Bm25Ranks,
    $RetrievalContext, $GenerationCheckpoint
)
foreach ($path in $pathsToCreate) {
    $parent = Split-Path -Parent $path
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

$rankFiles = @($AnchorRanks, $HarrierRanks, $Bm25Ranks)
$needRanks = $RebuildRetrieval -or $RebuildEmbeddingCaches
foreach ($rankFile in $rankFiles) {
    if (-not (Test-Path -LiteralPath $rankFile)) {
        $needRanks = $true
    }
}

if ($needRanks) {
    $anchorArgs = @(
        "-m", "src.evaluation.ensemble_fusion", "rank",
        "--name", "aiteamvn", "--embedding-model", $AnchorModel,
        "--embedding-cache", $AnchorEmbeddingCache, "--allow-encode",
        "--questions", $Questions, "--chunks", $Chunks,
        "--max-seq-length", "2048", "--top-n", "100",
        "--batch-size", "$EmbeddingBatchSize",
        "--question-batch-size", "$QuestionBatchSize",
        "--device", "cuda", "--out", $AnchorRanks
    )
    if ($RebuildEmbeddingCaches) { $anchorArgs += "--force-encode" }
    Invoke-PythonStep "1/6 Rank Task 2 with AITeamVN embedding" $anchorArgs

    $harrierArgs = @(
        "-m", "src.evaluation.ensemble_fusion", "rank",
        "--name", "harrier", "--embedding-model", $HarrierModel,
        "--trust-remote-code",
        "--query-instruct", "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question",
        "--embedding-cache", $HarrierEmbeddingCache, "--allow-encode",
        "--questions", $Questions, "--chunks", $Chunks,
        "--max-seq-length", "1024", "--top-n", "100",
        "--batch-size", "$EmbeddingBatchSize",
        "--question-batch-size", "$QuestionBatchSize",
        "--device", "cuda", "--out", $HarrierRanks
    )
    if ($RebuildEmbeddingCaches) { $harrierArgs += "--force-encode" }
    Invoke-PythonStep "2/6 Rank Task 2 with VietLegal Harrier" $harrierArgs

    Invoke-PythonStep "3/6 Rank Task 2 with BM25" @(
        "-m", "src.evaluation.ensemble_fusion", "rank-bm25",
        "--name", "bm25", "--questions", $Questions, "--chunks", $Chunks,
        "--top-n", "100", "--question-batch-size", "$QuestionBatchSize",
        "--out", $Bm25Ranks
    )
} else {
    Write-Host "Reusing Task 2 rank files. Use -RebuildRetrieval after input/model changes." -ForegroundColor Green
}

$contextWasBuilt = $false
if ($needRanks -or -not (Test-Path -LiteralPath $RetrievalContext)) {
    Invoke-PythonStep "4/6 Build ensemble LegalQA retrieval context" @(
        "-m", "src.evaluation.ensemble_context_for_task2",
        "--questions", $Questions, "--chunks", $Chunks,
        "--rank", "aiteamvn=$AnchorRanks",
        "--rank", "harrier=$HarrierRanks",
        "--rank", "bm25=$Bm25Ranks",
        "--weights", "harrier=0.2", "--weights", "bm25=0.1",
        "--default-top-n", "100", "--rrf-k", "60", "--top-k", "5",
        "--output", $RetrievalContext
    )
    $contextWasBuilt = $true
} else {
    Write-Host "Reusing ensemble retrieval context: $RetrievalContext" -ForegroundColor Green
}

$outputDirectory = Split-Path -Parent $Output
if (-not $outputDirectory) { $outputDirectory = "." }
$outputBaseName = [System.IO.Path]::GetFileNameWithoutExtension($Output)
$outputExtension = [System.IO.Path]::GetExtension($Output)
$rawOutput = Join-Path $outputDirectory ($outputBaseName + "_raw" + $outputExtension)
$diagnosticsOutput = Join-Path $outputDirectory "diagnostics_ensemble.json"
$postprocessReport = Join-Path $outputDirectory "postprocess_report_ensemble.json"

$runArgs = @(
    "-m", "src.evaluation.run_legalqa",
    "--questions", $Questions, "--chunks", $Chunks,
    "--llm-model", $LlmModel, "--output", $rawOutput,
    "--prompt-variant", $PromptVariant,
    "--retrieval-context-output", $RetrievalContext, "--reuse-retrieval-context",
    "--generation-checkpoint", $GenerationCheckpoint,
    "--diagnostics-output", $diagnosticsOutput,
    "--top-k", "5", "--max-documents", "5",
    "--max-chunks-per-document", "3", "--neighbor-window", "1",
    "--max-context-tokens", "4096", "--max-new-tokens", "384",
    "--answer-batch-size", "$AnswerBatchSize", "--llm-device", "cuda"
)

# Answers generated from an older context must not be resumed after retrieval changes.
if (-not $contextWasBuilt -and -not $RebuildGeneration -and (Test-Path -LiteralPath $GenerationCheckpoint)) {
    Write-Host "Resuming generated answers from: $GenerationCheckpoint" -ForegroundColor Green
    $runArgs += "--resume-generation"
}
if ($LocalFilesOnly) { $runArgs += "--local-files-only" }

Invoke-PythonStep "5/6 Generate LegalQA answers" $runArgs
Invoke-PythonStep "6/6 Postprocess Task 2 answers" @(
    "-m", "src.evaluation.postprocess_legalqa_submission",
    "--input", $rawOutput, "--output", $Output, "--report", $postprocessReport
)

$validateArgs = @(
    "-m", "src.evaluation.validate_legalqa_submission", "--predictions", $Output
)
if ($QuestionReferenceJson) { $validateArgs += @("--questions", $QuestionReferenceJson) }
Invoke-PythonStep "Validate Task 2 submission" $validateArgs

if ($Package) {
    $packageDir = Join-Path $outputDirectory "package"
    New-Item -ItemType Directory -Force -Path $packageDir | Out-Null
    Copy-Item -LiteralPath $Output -Destination (Join-Path $packageDir "submission.json") -Force
    $zipPath = Join-Path $outputDirectory "submission.zip"
    Compress-Archive -Path (Join-Path $packageDir "submission.json") -DestinationPath $zipPath -Force
    Write-Host "Created $zipPath" -ForegroundColor Green
}

Write-Host "`nTask 2 pipeline completed: $Output" -ForegroundColor Green
