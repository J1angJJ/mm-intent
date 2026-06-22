Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $projectRoot = (Get-Location).Path
}
else {
    $projectRoot = $PSScriptRoot
}
Set-Location $projectRoot

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
chcp.com 65001 > $null
[Console]::InputEncoding = $Utf8NoBom
[Console]::OutputEncoding = $Utf8NoBom
$OutputEncoding = $Utf8NoBom
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$utf8Runner = Join-Path $projectRoot "run_utf8_child.py"
$logDir = Join-Path $projectRoot "logs_raw_missing_rerun"
$cleanClip = Join-Path $projectRoot "outputs\raw_feature_cache\clean_seed42"
$cleanHand = Join-Path $projectRoot "outputs\raw_feature_cache\clean_seed42_hand_geometry"

foreach ($requiredPath in @($python, $utf8Runner, $cleanClip, $cleanHand)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Required path not found: $requiredPath"
    }
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [string] $Name,
        [string[]] $Arguments
    )

    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan
    $logPath = Join-Path $logDir ($Name + ".log")
    $runnerArguments = @(
        "-X", "utf8", "-u", $utf8Runner,
        "--log", $logPath,
        "--", $python, "-X", "utf8", "-u"
    )
    $runnerArguments += $Arguments
    & $python @runnerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE. Log: $logPath"
    }
}

foreach ($variableName in @(
    "SMART_AR_MISSING_MODALITIES",
    "SMART_AR_NOISE_MODALITY",
    "SMART_AR_NOISE_LEVEL",
    "SMART_AR_SKIP_TEST_EVAL",
    "MM_INTENT_RAW_NOISE_MODALITY",
    "MM_INTENT_RAW_NOISE_LEVEL",
    "MM_INTENT_GESTURE_FEATURE_DIR",
    "MM_INTENT_GESTURE_FEAT_DIM"
)) {
    Remove-Item "Env:$variableName" -ErrorAction SilentlyContinue
}

# Training uses --skip-test-eval internally; run_missing_experiments.py then runs
# one independent code/test.py pass. This avoids duplicate test/subset evaluation.
Invoke-Step -Name "baseline_raw_missing" -Arguments @(
    "code\run_missing_experiments.py",
    "--model", "baseline",
    "--output-model-name", "baseline",
    "--input-mode", "raw",
    "--gesture-representation", "clip",
    "--base-feature-dir", $cleanClip,
    "--seed", "123",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "10",
    "--max-missing", "2",
    "--skip-existing",
    "--execute"
)

Invoke-Step -Name "improved_raw_missing" -Arguments @(
    "code\run_missing_experiments.py",
    "--model", "improved",
    "--output-model-name", "improved",
    "--input-mode", "raw",
    "--gesture-representation", "clip",
    "--base-feature-dir", $cleanClip,
    "--seed", "42",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--max-missing", "2",
    "--skip-existing",
    "--execute"
)

Invoke-Step -Name "hand_geometry_raw_missing" -Arguments @(
    "code\run_missing_experiments.py",
    "--model", "improved",
    "--output-model-name", "hand_geometry",
    "--input-mode", "raw",
    "--gesture-representation", "hand_geometry",
    "--base-feature-dir", $cleanHand,
    "--seed", "42",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--max-missing", "2",
    "--skip-existing",
    "--execute"
)

Write-Host ""
Write-Host "===== Final completeness check =====" -ForegroundColor Cyan
foreach ($name in @("baseline", "improved", "hand_geometry")) {
    $root = Join-Path $projectRoot "outputs\raw_missing_experiments\$name"
    $trainCount = if (Test-Path $root) {
        @(Get-ChildItem $root -Recurse -File -Filter "metrics.json").Count
    }
    else { 0 }
    $testCount = if (Test-Path $root) {
        @(Get-ChildItem $root -Recurse -File -Filter "independent_test_metrics.json").Count
    }
    else { 0 }

    Write-Host "$name : train=$trainCount/15 independent_test=$testCount/15"
    if ($trainCount -ne 15 -or $testCount -ne 15) {
        throw "$name raw-missing results are incomplete"
    }
}

Write-Host "ALL REQUIRED RAW-MISSING RUNS COMPLETED" -ForegroundColor Green
