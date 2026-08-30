<#
.SYNOPSIS
Run A1 end-to-end: fixed validation split, prompt comparison, selection and final Task 2 submission.

.DESCRIPTION
This script intentionally keeps retrieval context, LLM, context budget and greedy decoding fixed.
Only --prompt-variant changes during validation.  It then selects the best eligible variant
by METEOR (without a ROUGE-L decrease by default), reruns public generation with that prompt,
and writes a reproducibility log.
#>
param(
    [string]$TrainQuestions = "data/processed/train_2_preprocessed.jsonl",
    [string]$Chunks = "data/processed/chunks_corpus.jsonl",
    [string]$ExperimentDir = "outputs/task2/a1",
    [int]$ValidationSize = 200,
    [int]$Seed = 20260828,
    [string[]]$PromptVariants = @("baseline", "answer_first", "legal_structured", "grounded_concise"),
    [ValidateSet("baseline", "answer_first", "legal_structured", "grounded_concise")]
    [string]$BaselineVariant = "baseline",
    [double]$MaxRougeDrop = 0.0,
    [string]$AnchorModel = "AITeamVN/Vietnamese_Embedding",
    [string]$HarrierModel = "mainguyen9/vietlegal-harrier-0.6b",
    [string]$LlmModel = "AITeamVN/Vi-Qwen2-1.5B-RAG",
    [string]$AnchorEmbeddingCache = "outputs/cache/embcompare_aiteamvn.npz",
    [string]$HarrierEmbeddingCache = "outputs/cache/embcompare_vietlegal_harrier.npz",
    [string]$PublicQuestions = "data/processed/public_test_2_preprocessed.jsonl",
    [string]$PublicQuestionReferenceJson = "data/raw/public-official-2.json",
    [string]$PublicOutput = "outputs/task2/submission_a1_best_prompt.json",
    [int]$EmbeddingBatchSize = 64,
    [int]$QuestionBatchSize = 64,
    [int]$AnswerBatchSize = 1,
    [string]$PythonExecutable = "",
    [switch]$RebuildValidationSplit,
    [switch]$RebuildValidationRetrieval,
    [switch]$ResumeValidationGeneration,
    [switch]$LocalFilesOnly,
    [switch]$SkipPackage
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

function Test-RetrievalDependencies {
    Write-Host "Checking sentence-transformers in the selected environment..." -ForegroundColor DarkGray
    & $PythonExecutable -c "import sys; import sentence_transformers; print('sentence-transformers', sentence_transformers.__version__); print('Python', sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Cannot import sentence-transformers with: $PythonExecutable`n" +
            "Install the missing retrieval dependency into this exact environment, then rerun:`n" +
            "  & '$PythonExecutable' -m pip install 'sentence-transformers[train]>=3.0.0'`n" +
            "This does not delete models, embedding caches, validation files, or checkpoints."
        )
    }
}

Test-RetrievalDependencies

$allowedVariants = @("baseline", "answer_first", "legal_structured", "grounded_concise")
if ($PromptVariants.Count -lt 2) { throw "Provide at least baseline and one candidate in -PromptVariants." }
if ($PromptVariants -notcontains $BaselineVariant) { throw "-PromptVariants must include -BaselineVariant." }
foreach ($variant in $PromptVariants) {
    if ($allowedVariants -notcontains $variant) {
        throw "Unknown prompt variant '$variant'. Allowed: $($allowedVariants -join ', ')"
    }
}
if ($MaxRougeDrop -lt 0) { throw "-MaxRougeDrop must be >= 0." }

function Invoke-PythonStep {
    param([string]$Title, [string[]]$Arguments)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $PythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Title failed with exit code $LASTEXITCODE" }
}

function Require-Path {
    param([string]$Path, [string]$Description)
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Description not found: $Path" }
}

Require-Path "requirements.txt" "Repository root marker"
Require-Path $TrainQuestions "Answered Task 2 training data"
Require-Path $Chunks "Shared chunks corpus"
Require-Path $AnchorEmbeddingCache "AITeamVN embedding cache"
Require-Path $HarrierEmbeddingCache "Harrier embedding cache"
Require-Path $PublicQuestions "Task 2 public questions"

New-Item -ItemType Directory -Force -Path $ExperimentDir | Out-Null
$valQuestions = Join-Path $ExperimentDir "val_questions.jsonl"
$valReferences = Join-Path $ExperimentDir "val_references.json"
$valManifest = Join-Path $ExperimentDir "val_split_manifest.json"
$anchorRanks = Join-Path $ExperimentDir "ranks_aiteamvn.npz"
$harrierRanks = Join-Path $ExperimentDir "ranks_harrier.npz"
$bm25Ranks = Join-Path $ExperimentDir "ranks_bm25.npz"
$valContext = Join-Path $ExperimentDir "retrieval_context_ensemble.json"

if ($RebuildValidationSplit -or -not (Test-Path -LiteralPath $valQuestions)) {
    $splitArgs = @(
        "-m", "src.evaluation.prepare_legalqa_prompt_validation",
        "--source", $TrainQuestions,
        "--questions-output", $valQuestions,
        "--references-output", $valReferences,
        "--manifest-output", $valManifest,
        "--size", "$ValidationSize", "--seed", "$Seed"
    )
    if ($RebuildValidationSplit) { $splitArgs += "--overwrite" }
    Invoke-PythonStep "A1.1 Create fixed answered validation split" $splitArgs
} else {
    Require-Path $valReferences "Validation references"
    Require-Path $valManifest "Validation split manifest"
    Write-Host "Reusing fixed validation split: $valQuestions" -ForegroundColor Green
}

$rankFiles = @($anchorRanks, $harrierRanks, $bm25Ranks)
$needRanks = $RebuildValidationSplit -or $RebuildValidationRetrieval
foreach ($rankFile in $rankFiles) {
    if (-not (Test-Path -LiteralPath $rankFile)) { $needRanks = $true }
}

if ($needRanks) {
    Invoke-PythonStep "A1.2 Rank validation questions with AITeamVN cache" @(
        "-m", "src.evaluation.ensemble_fusion", "rank",
        "--name", "aiteamvn", "--embedding-model", $AnchorModel,
        "--embedding-cache", $AnchorEmbeddingCache,
        "--questions", $valQuestions, "--chunks", $Chunks,
        "--max-seq-length", "2048", "--top-n", "100",
        "--batch-size", "$EmbeddingBatchSize", "--question-batch-size", "$QuestionBatchSize",
        "--device", "cuda", "--out", $anchorRanks
    )
    Invoke-PythonStep "A1.3 Rank validation questions with Harrier cache" @(
        "-m", "src.evaluation.ensemble_fusion", "rank",
        "--name", "harrier", "--embedding-model", $HarrierModel,
        "--trust-remote-code",
        "--query-instruct", "Given a Vietnamese legal question, retrieve relevant legal passages that answer the question",
        "--embedding-cache", $HarrierEmbeddingCache,
        "--questions", $valQuestions, "--chunks", $Chunks,
        "--max-seq-length", "1024", "--top-n", "100",
        "--batch-size", "$EmbeddingBatchSize", "--question-batch-size", "$QuestionBatchSize",
        "--device", "cuda", "--out", $harrierRanks
    )
    Invoke-PythonStep "A1.4 Rank validation questions with BM25" @(
        "-m", "src.evaluation.ensemble_fusion", "rank-bm25",
        "--name", "bm25", "--questions", $valQuestions, "--chunks", $Chunks,
        "--top-n", "100", "--question-batch-size", "$QuestionBatchSize", "--out", $bm25Ranks
    )
} else {
    Write-Host "Reusing validation rank files; corpus embeddings will not be re-encoded." -ForegroundColor Green
}

if ($needRanks -or -not (Test-Path -LiteralPath $valContext)) {
    Invoke-PythonStep "A1.5 Build fixed ensemble retrieval context" @(
        "-m", "src.evaluation.ensemble_context_for_task2",
        "--questions", $valQuestions, "--chunks", $Chunks,
        "--rank", "aiteamvn=$anchorRanks", "--rank", "harrier=$harrierRanks", "--rank", "bm25=$bm25Ranks",
        "--weights", "harrier=0.2", "--weights", "bm25=0.1",
        "--default-top-n", "100", "--rrf-k", "60", "--top-k", "5", "--output", $valContext
    )
} else {
    Write-Host "Reusing validation retrieval context: $valContext" -ForegroundColor Green
}

$reportArgs = @()
foreach ($variant in $PromptVariants) {
    $rawOutput = Join-Path $ExperimentDir "submission_${variant}_raw.json"
    $finalOutput = Join-Path $ExperimentDir "submission_${variant}.json"
    $checkpoint = Join-Path $ExperimentDir "generation_checkpoint_${variant}.jsonl"
    $diagnostics = Join-Path $ExperimentDir "diagnostics_${variant}.json"
    $postprocessReport = Join-Path $ExperimentDir "postprocess_${variant}.json"
    $metricReport = Join-Path $ExperimentDir "metrics_${variant}.json"

    $generationArgs = @(
        "-m", "src.evaluation.run_legalqa",
        "--questions", $valQuestions, "--chunks", $Chunks,
        "--llm-model", $LlmModel, "--prompt-variant", $variant,
        "--output", $rawOutput,
        "--retrieval-context-output", $valContext, "--reuse-retrieval-context",
        "--generation-checkpoint", $checkpoint, "--diagnostics-output", $diagnostics,
        "--top-k", "5", "--max-documents", "5", "--max-chunks-per-document", "3",
        "--neighbor-window", "1", "--max-context-tokens", "4096", "--max-new-tokens", "384",
        "--answer-batch-size", "$AnswerBatchSize", "--llm-device", "cuda"
    )
    if ($LocalFilesOnly) { $generationArgs += "--local-files-only" }
    if ($ResumeValidationGeneration -and (Test-Path -LiteralPath $checkpoint)) {
        $generationArgs += "--resume-generation"
    }
    Invoke-PythonStep "A1.6 Generate validation answers ($variant)" $generationArgs
    Invoke-PythonStep "A1.7 Postprocess answers ($variant)" @(
        "-m", "src.evaluation.postprocess_legalqa_submission",
        "--input", $rawOutput, "--output", $finalOutput, "--report", $postprocessReport
    )
    Invoke-PythonStep "A1.8 Evaluate prompt ($variant)" @(
        "-m", "src.evaluation.evaluate_legalqa",
        "--references", $valReferences, "--predictions", $finalOutput, "--report", $metricReport
    )
    $reportArgs += @("--report", "${variant}=${metricReport}")
}

$selectionPath = Join-Path $ExperimentDir "prompt_selection.json"
$selectionArgs = @(
    "-m", "src.evaluation.select_legalqa_prompt_variant",
    "--baseline", $BaselineVariant, "--max-rouge-drop", "$MaxRougeDrop",
    "--output", $selectionPath
) + $reportArgs
Invoke-PythonStep "A1.9 Select the best prompt variant" $selectionArgs

$selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$bestVariant = [string]$selection.best_variant
Write-Host "`nBest prompt variant: $bestVariant" -ForegroundColor Green

$productionScript = "scripts/run_pipeline_task2.ps1"
Require-Path $productionScript "Production Task 2 wrapper"
$productionArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", $productionScript,
    "-Questions", $PublicQuestions, "-QuestionReferenceJson", $PublicQuestionReferenceJson,
    "-Chunks", $Chunks, "-Output", $PublicOutput,
    "-AnchorModel", $AnchorModel, "-HarrierModel", $HarrierModel, "-LlmModel", $LlmModel,
    "-AnchorEmbeddingCache", $AnchorEmbeddingCache, "-HarrierEmbeddingCache", $HarrierEmbeddingCache,
    "-PromptVariant", $bestVariant, "-AnswerBatchSize", "$AnswerBatchSize",
    "-PythonExecutable", $PythonExecutable, "-RebuildGeneration"
)
if ($LocalFilesOnly) { $productionArgs += "-LocalFilesOnly" }
if (-not $SkipPackage) { $productionArgs += "-Package" }
Write-Host "`n=== A1.10 Generate public submission using '$bestVariant' ===" -ForegroundColor Cyan
& powershell @productionArgs
if ($LASTEXITCODE -ne 0) { throw "Final public generation failed with exit code $LASTEXITCODE" }
Require-Path $PublicOutput "Final public submission"

$publicDir = Split-Path -Parent $PublicOutput
if (-not $publicDir) { $publicDir = "." }
$publicBase = [System.IO.Path]::GetFileNameWithoutExtension($PublicOutput)
$runLog = [ordered]@{
    completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    experiment = "A1_prompt_engineering"
    best_variant = $bestVariant
    selection = $selection
    fixed_settings = [ordered]@{
        llm_model = $LlmModel
        decoding = "greedy (do_sample=false)"
        retrieval = "AITeamVN=1.0, Harrier=0.2, BM25=0.1, RRF k=60, top_k=5"
        context = "max_documents=5, max_chunks_per_document=3, neighbor_window=1, max_context_tokens=4096"
        max_new_tokens = 384
        validation_size = $ValidationSize
        validation_seed = $Seed
    }
    artifacts = [ordered]@{
        validation_manifest = $valManifest
        validation_context = $valContext
        prompt_selection = $selectionPath
        final_submission = $PublicOutput
        final_submission_zip = (Join-Path $publicDir "submission.zip")
        prompt_source_sha256 = (Get-FileHash -Algorithm SHA256 "src/qa/prompts.py").Hash.ToLower()
        production_wrapper_sha256 = (Get-FileHash -Algorithm SHA256 $productionScript).Hash.ToLower()
    }
}
$runLogPath = Join-Path $ExperimentDir "best_pipeline_run.json"
$runLog | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $runLogPath -Encoding utf8
Write-Host "`nA1 completed. Best-run log: $runLogPath" -ForegroundColor Green
Write-Host "Submission: $PublicOutput" -ForegroundColor Green
