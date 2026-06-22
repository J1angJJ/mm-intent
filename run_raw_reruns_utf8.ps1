Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ============================================================
# 0. Project root and UTF-8 configuration
# ============================================================

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

# ============================================================
# 1. Paths
# ============================================================

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$utf8Runner = Join-Path $projectRoot "run_utf8_child.py"
$baseFeatures = Join-Path $projectRoot "dataset\AR_Data_Process3.0\data_full"
$logDir = Join-Path $projectRoot "logs_raw_rerun"

if (-not (Test-Path $python)) {
    throw "Python interpreter not found: $python"
}

if (-not (Test-Path $utf8Runner)) {
    throw "UTF-8 runner not found: $utf8Runner"
}

if (-not (Test-Path $baseFeatures)) {
    throw "Base feature directory not found: $baseFeatures"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "Project root : $projectRoot"
Write-Host "Python       : $python"
Write-Host "Base features: $baseFeatures"
Write-Host "Log directory: $logDir"

# ============================================================
# 2. Clear experiment-related environment variables
# ============================================================

$variablesToClear = @(
    "SMART_AR_NOISE_MODALITY",
    "SMART_AR_NOISE_LEVEL",
    "SMART_AR_MISSING_MODALITIES",
    "SMART_AR_TEST_VIDEO_NAMES",
    "SMART_AR_SKIP_TEST_EVAL",
    "SMART_AR_RANDOM_SEED",
    "MM_INTENT_RAW_NOISE_MODALITY",
    "MM_INTENT_RAW_NOISE_LEVEL",
    "MM_INTENT_GESTURE_FEATURE_DIR",
    "MM_INTENT_GESTURE_FEAT_DIM"
)

foreach ($variableName in $variablesToClear) {
    Remove-Item "Env:$variableName" -ErrorAction SilentlyContinue
}

# ============================================================
# 3. UTF-8-safe process runner
# ============================================================

function Invoke-Step {
    param(
        [string] $Name,
        [string[]] $Arguments
    )

    Write-Host ""
    Write-Host "===== $Name =====" -ForegroundColor Cyan

    $logPath = Join-Path $logDir ($Name + ".log")

    if (Test-Path $logPath) {
        Remove-Item $logPath -Force
    }

    $runnerArguments = @(
        "-X",
        "utf8",
        "-u",
        $utf8Runner,
        "--log",
        $logPath,
        "--",
        $python,
        "-X",
        "utf8",
        "-u"
    )

    $runnerArguments += $Arguments

    & $python @runnerArguments
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode. Log: $logPath"
    }

    Write-Host "Completed: $Name" -ForegroundColor Green
}

function Invoke-StepIfMissing {
    param(
        [string] $Name,
        [string[]] $RequiredPaths,
        [string[]] $Arguments
    )

    $missingPaths = @($RequiredPaths | Where-Object { -not (Test-Path $_) })
    if ($missingPaths.Count -eq 0) {
        Write-Host "[skip-existing] $Name" -ForegroundColor DarkGreen
        return
    }
    Invoke-Step -Name $Name -Arguments $Arguments
}

# ============================================================
# 4. Environment preflight
# ============================================================

Invoke-Step -Name "environment_preflight" -Arguments @(
    "-c",
    "import locale,sys; print('sys.executable:',sys.executable); print('stdout:',sys.stdout.encoding); print('locale:',locale.getpreferredencoding(False)); print('中文测试：原始数据端到端实验开始')"
)

# ============================================================
# 5. Baseline: clean raw-data end-to-end
# ============================================================

Invoke-StepIfMissing -Name "baseline_clean_train" `
    -RequiredPaths @("outputs\baseline_raw_end_to_end\metrics.json") `
    -Arguments @(
    "code\train.py",
    "--model", "baseline",
    "--input-mode", "raw",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
    "--seed", "123",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "10",
    "--output-dir", "outputs\baseline_raw_end_to_end"
)

Invoke-StepIfMissing -Name "baseline_clean_test" `
    -RequiredPaths @("outputs\baseline_raw_end_to_end\independent_test_metrics.json") `
    -Arguments @(
    "code\test.py",
    "--model", "baseline",
    "--input-mode", "raw",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
    "--output-dir", "outputs\baseline_raw_end_to_end"
)

# ============================================================
# 6. Given Improved: clean raw-data end-to-end
# ============================================================

Invoke-StepIfMissing -Name "improved_clean_train" `
    -RequiredPaths @("outputs\improved_raw_end_to_end\metrics.json") `
    -Arguments @(
    "code\train.py",
    "--model", "improved",
    "--input-mode", "raw",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
    "--seed", "42",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--output-dir", "outputs\improved_raw_end_to_end"
)

Invoke-StepIfMissing -Name "improved_clean_test" `
    -RequiredPaths @("outputs\improved_raw_end_to_end\independent_test_metrics.json") `
    -Arguments @(
    "code\test.py",
    "--model", "improved",
    "--input-mode", "raw",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
    "--output-dir", "outputs\improved_raw_end_to_end"
)

# ============================================================
# 7. Hand Geometry: clean raw-data end-to-end
# ============================================================

Invoke-StepIfMissing -Name "hand_geometry_clean_train" `
    -RequiredPaths @("outputs\hand_geometry_raw_end_to_end\metrics.json") `
    -Arguments @(
    "code\train.py",
    "--model", "improved",
    "--input-mode", "raw",
    "--gesture-representation", "hand_geometry",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42_hand_geometry",
    "--base-feature-dir", $baseFeatures,
    "--seed", "42",
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--output-dir", "outputs\hand_geometry_raw_end_to_end"
)

Invoke-StepIfMissing -Name "hand_geometry_clean_test" `
    -RequiredPaths @("outputs\hand_geometry_raw_end_to_end\independent_test_metrics.json") `
    -Arguments @(
    "code\test.py",
    "--model", "improved",
    "--input-mode", "raw",
    "--gesture-representation", "hand_geometry",
    "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42_hand_geometry",
    "--base-feature-dir", $baseFeatures,
    "--output-dir", "outputs\hand_geometry_raw_end_to_end"
)

# ============================================================
# 8. Decode Gesture once and precompute all three noise levels
# ============================================================

Invoke-Step -Name "precompute_clip_gesture_noise_levels" -Arguments @(
    "code\precompute_gesture_noise_levels.py",
    "--representation", "clip",
    "--levels", "0.2", "0.4", "0.6",
    "--noise-seed", "42",
    "--base-feature-dir", $baseFeatures,
    "--clean-scene-cache", "outputs\raw_feature_cache\clean_seed42\scene_features"
)

Invoke-Step -Name "precompute_hand_geometry_noise_levels" -Arguments @(
    "code\precompute_gesture_noise_levels.py",
    "--representation", "hand_geometry",
    "--levels", "0.2", "0.4", "0.6",
    "--noise-seed", "42",
    "--base-feature-dir", $baseFeatures,
    "--clean-scene-cache", "outputs\raw_feature_cache\clean_seed42\scene_features"
)

# ============================================================
# 9. Baseline: 15 raw-modality noise training runs
# ============================================================

$env:SMART_AR_RANDOM_SEED = "123"

Invoke-Step -Name "baseline_raw_noise_train" -Arguments @(
    "code\run_noise_experiments.py",
    "--model", "baseline",
    "--output-model-name", "baseline",
    "--base-feature-dir", $baseFeatures,
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "10",
    "--skip-existing",
    "--execute"
)

# ============================================================
# 10. Reuse Baseline raw perturbations for Hand Geometry
# ============================================================

Invoke-Step -Name "clone_hand_geometry_noise_caches" -Arguments @(
    "code\clone_hand_geometry_noise_caches.py",
    "--levels", "0.2", "0.4", "0.6",
    "--modalities", "imu", "audio", "text", "scene",
    "--noise-seed", "42",
    "--clean-feature-dir", $baseFeatures
)

# ============================================================
# 11. Given Improved: 15 raw-modality noise training runs
# ============================================================

$env:SMART_AR_RANDOM_SEED = "42"

Invoke-Step -Name "improved_raw_noise_train" -Arguments @(
    "code\run_noise_experiments.py",
    "--model", "improved",
    "--output-model-name", "improved",
    "--base-feature-dir", $baseFeatures,
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--skip-existing",
    "--execute"
)

# ============================================================
# 12. Hand Geometry: 15 raw-modality noise training runs
# ============================================================

$env:SMART_AR_RANDOM_SEED = "42"

Invoke-Step -Name "hand_geometry_raw_noise_train" -Arguments @(
    "code\run_noise_experiments.py",
    "--model", "improved",
    "--output-model-name", "hand_geometry",
    "--gesture-representation", "hand_geometry",
    "--base-feature-dir", $baseFeatures,
    "--noise-seed", "42",
    "--epochs", "100",
    "--patience", "4",
    "--skip-existing",
    "--execute"
)

# ============================================================
# 13. Independent tests for all 45 noise checkpoints
# ============================================================

$configs = @(
    @{
        Name = "baseline"
        Model = "baseline"
        Representation = "clip"
    },
    @{
        Name = "improved"
        Model = "improved"
        Representation = "clip"
    },
    @{
        Name = "hand_geometry"
        Model = "improved"
        Representation = "hand_geometry"
    }
)

$modalities = @("imu", "gesture", "audio", "text", "scene")
$percentages = @(20, 40, 60)

foreach ($config in $configs) {
    foreach ($modality in $modalities) {
        foreach ($percent in $percentages) {
            $level = ($percent / 100.0).ToString(
                [System.Globalization.CultureInfo]::InvariantCulture
            )

            if ($config.Representation -eq "hand_geometry") {
                $suffix = "_hand_geometry"
            }
            else {
                $suffix = ""
            }

            $cacheDir = "outputs\raw_feature_cache\${modality}_noise_${percent}_seed42${suffix}"
            $outputDir = "outputs\raw_noise_experiments\$($config.Name)\${modality}_noise_$percent"
            $testName = "$($config.Name)_${modality}_noise_${percent}_test"

            Invoke-StepIfMissing -Name $testName `
                -RequiredPaths @("$outputDir\independent_test_metrics.json") `
                -Arguments @(
                "code\test.py",
                "--model", $config.Model,
                "--input-mode", "raw",
                "--gesture-representation", $config.Representation,
                "--noise-modality", $modality,
                "--noise-level", $level,
                "--noise-seed", "42",
                "--raw-cache-dir", $cacheDir,
                "--base-feature-dir", $baseFeatures,
                "--output-dir", $outputDir
            )
        }
    }
}

# ============================================================
# 14. Final completeness check
# ============================================================

Write-Host ""
Write-Host "===== Final completeness check =====" -ForegroundColor Cyan

foreach ($name in @("baseline", "improved", "hand_geometry")) {
    $root = Join-Path $projectRoot "outputs\raw_noise_experiments\$name"

    if (-not (Test-Path $root)) {
        throw "Missing result directory: $root"
    }

    $trainCount = @(
        Get-ChildItem $root -Recurse -File -Filter "metrics.json"
    ).Count

    $testCount = @(
        Get-ChildItem $root -Recurse -File -Filter "independent_test_metrics.json"
    ).Count

    Write-Host "$name : train=$trainCount/15 independent_test=$testCount/15"

    if ($trainCount -ne 15) {
        throw "$name training results are incomplete: $trainCount/15"
    }

    if ($testCount -ne 15) {
        throw "$name independent-test results are incomplete: $testCount/15"
    }
}

$cleanOutputs = @(
    @{
        Name = "baseline"
        Directory = "outputs\baseline_raw_end_to_end"
    },
    @{
        Name = "improved"
        Directory = "outputs\improved_raw_end_to_end"
    },
    @{
        Name = "hand_geometry"
        Directory = "outputs\hand_geometry_raw_end_to_end"
    }
)

foreach ($cleanOutput in $cleanOutputs) {
    $directory = Join-Path $projectRoot $cleanOutput.Directory
    $trainingMetrics = Join-Path $directory "metrics.json"
    $independentMetrics = Join-Path $directory "independent_test_metrics.json"

    if (-not (Test-Path $trainingMetrics)) {
        throw "Missing clean training metrics: $trainingMetrics"
    }

    if (-not (Test-Path $independentMetrics)) {
        throw "Missing clean independent-test metrics: $independentMetrics"
    }

    Write-Host "$($cleanOutput.Name) clean run: training metrics OK, independent test OK"
}

Write-Host ""
Write-Host "ALL REQUIRED RAW RERUNS COMPLETED" -ForegroundColor Green
