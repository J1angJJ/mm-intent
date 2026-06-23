param(
    [switch] $Resume,
    [switch] $RebuildRawFeatureCache
)

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
$global:OutputEncoding = $Utf8NoBom

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

# All three methods use the same seed set in the repeated-seed
# generalization experiments.
$generalizationSeeds = @("7", "42", "123")
$dateHoldoutSeed = "42"

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$runMode = if ($Resume) { "resume" } else { "fresh" }

# ============================================================
# 1. Paths
# ============================================================

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$baseFeatures = Join-Path $projectRoot "dataset\AR_Data_Process3.0\data_full"
$outputRoot = Join-Path $projectRoot "outputs"
$rawCacheRoot = Join-Path $outputRoot "raw_feature_cache"
$cleanClip = Join-Path $rawCacheRoot "clean_seed42"
$cleanHand = Join-Path $rawCacheRoot "clean_seed42_hand_geometry"

$handGeometryFeatureDir = Join-Path $baseFeatures "hand_geometry_features"
$sceneGestureFeatureDir = Join-Path $baseFeatures "strong_gesture_features"

$logRoot = Join-Path $projectRoot "logs_all_readme_experiments"
$logDir = Join-Path $logRoot $runId

if ([string]::IsNullOrWhiteSpace($env:TEMP)) {
    $helperRoot = Join-Path $projectRoot ".tmp"
}
else {
    $helperRoot = Join-Path $env:TEMP "mm_intent_all_readme_$PID"
}

$utf8Runner = Join-Path $helperRoot "run_utf8_child.py"

$env:MM_INTENT_PROCESSED_DATA_DIR = $baseFeatures
$env:MM_INTENT_OUTPUT_DIR = $outputRoot

if ([string]::IsNullOrWhiteSpace($env:HF_HOME)) {
    $env:HF_HOME = Join-Path $env:USERPROFILE ".cache\huggingface"
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $helperRoot | Out-Null

# ============================================================
# 2. Embedded UTF-8 child-process runner
#    The distributed workflow remains a single PowerShell file.
# ============================================================

$runnerSource = @'
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )


def parse_args(argv: list[str]) -> tuple[Path, list[str]]:
    try:
        log_index = argv.index("--log")
        separator_index = argv.index("--")
    except ValueError as exc:
        raise ValueError(
            "Usage: run_utf8_child.py --log LOG_FILE -- COMMAND [ARGS...]"
        ) from exc

    if log_index + 1 >= separator_index:
        raise ValueError("Missing log file path after --log.")

    log_path = Path(argv[log_index + 1]).resolve()
    command = argv[separator_index + 1:]

    if not command:
        raise ValueError("Missing child command after --.")

    return log_path, command


def main() -> int:
    configure_stdio()

    try:
        log_path, command = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    log_path.parent.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUNBUFFERED"] = "1"

    header = (
        f"COMMAND: {subprocess.list2cmdline(command)}\n"
        f"WORKING DIRECTORY: {Path.cwd()}\n"
        "TEXT ENCODING: UTF-8\n\n"
    )

    process: subprocess.Popen[str] | None = None

    with log_path.open(
        "w",
        encoding="utf-8",
        newline="",
        buffering=1,
    ) as log_file:
        sys.stdout.write(header)
        log_file.write(header)

        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout is None:
                raise RuntimeError("Unable to capture child-process output.")

            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)

            return_code = process.wait()

        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            message = "\nInterrupted by user.\n"
            sys.stdout.write(message)
            log_file.write(message)
            return 130

        footer = f"\nEXIT CODE: {return_code}\n"
        sys.stdout.write(footer)
        log_file.write(footer)

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
'@

[System.IO.File]::WriteAllText(
    $utf8Runner,
    $runnerSource,
    $Utf8NoBom
)

# ============================================================
# 3. Utility functions
# ============================================================

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
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

function Invoke-StepMaybe {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [string[]] $RequiredPaths,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    if ($Resume) {
        $missingPaths = @(
            $RequiredPaths |
                Where-Object { -not (Test-Path $_) }
        )

        if ($missingPaths.Count -eq 0) {
            Write-Host "[resume-skip-complete] $Name" -ForegroundColor DarkGreen
            return
        }
    }

    Invoke-Step -Name $Name -Arguments $Arguments
}

function Move-ExistingToBackup {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RelativePath,

        [Parameter(Mandatory = $true)]
        [string] $BackupRoot
    )

    $source = Join-Path $projectRoot $RelativePath

    if (-not (Test-Path $source)) {
        return
    }

    $safeName = $RelativePath -replace '[\\/:*?"<>|\s]+', '__'
    $destination = Join-Path $BackupRoot $safeName

    Move-Item -LiteralPath $source -Destination $destination
    Write-Host "Archived: $source -> $destination"
}

function Sync-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Source,

        [Parameter(Mandatory = $true)]
        [string] $Destination
    )

    if (-not (Test-Path $Source)) {
        throw "Cannot mirror missing source directory: $Source"
    }

    if (Test-Path $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    Get-ChildItem -LiteralPath $Source -Force |
        Copy-Item -Destination $Destination -Recurse -Force

    Write-Host "Mirrored: $Source -> $Destination"
}

function Publish-MetricsTree {
    param(
        [Parameter(Mandatory = $true)]
        [string] $SourceRoot,

        [Parameter(Mandatory = $true)]
        [string] $DestinationRoot
    )

    if (-not (Test-Path $SourceRoot)) {
        throw "Metrics source directory not found: $SourceRoot"
    }

    if (Test-Path $DestinationRoot) {
        Remove-Item -LiteralPath $DestinationRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    $sourceFull = (Resolve-Path $SourceRoot).Path

    foreach ($file in Get-ChildItem $SourceRoot -Recurse -File -Filter "metrics.json") {
        $relative = $file.FullName.Substring($sourceFull.Length)
        $relative = $relative.TrimStart([char[]]"\/")
        $destination = Join-Path $DestinationRoot $relative
        $destinationParent = Split-Path $destination -Parent

        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }

    Write-Host "Published lightweight metrics tree: $SourceRoot -> $DestinationRoot"
}

function Count-Files {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Root,

        [Parameter(Mandatory = $true)]
        [string] $Filter
    )

    if (-not (Test-Path $Root)) {
        return 0
    }

    return @(
        Get-ChildItem $Root -Recurse -File -Filter $Filter
    ).Count
}

function Clear-ExperimentEnvironment {
    foreach ($variableName in @(
        "SMART_AR_NOISE_MODALITY",
        "SMART_AR_NOISE_LEVEL",
        "SMART_AR_MISSING_MODALITIES",
        "SMART_AR_TEST_VIDEO_NAMES",
        "SMART_AR_SKIP_TEST_EVAL",
        "SMART_AR_RANDOM_SEED",
        "SMART_AR_MODEL_OUTPUT_DIR",
        "MM_INTENT_RAW_NOISE_MODALITY",
        "MM_INTENT_RAW_NOISE_LEVEL",
        "MM_INTENT_GESTURE_FEATURE_DIR",
        "MM_INTENT_GESTURE_FEAT_DIM"
    )) {
        Remove-Item "Env:$variableName" -ErrorAction SilentlyContinue
    }
}

# ============================================================
# 4. Execution
# ============================================================

try {
    # --------------------------------------------------------
    # 4.1 Preflight
    # --------------------------------------------------------

    $requiredPaths = @(
        $python,
        $baseFeatures,
        $handGeometryFeatureDir,
        $sceneGestureFeatureDir,
        (Join-Path $projectRoot "code\train.py"),
        (Join-Path $projectRoot "code\test.py"),
        (Join-Path $projectRoot "code\run_noise_experiments.py"),
        (Join-Path $projectRoot "code\run_missing_experiments.py"),
        (Join-Path $projectRoot "code\run_generalization_experiments.py"),
        (Join-Path $projectRoot "code\run_factorized_full_suite.py"),
        (Join-Path $projectRoot "code\evaluate_factorized_head_fusion.py"),
        (Join-Path $projectRoot "code\visualize_wjy_results.py"),
        (Join-Path $projectRoot "code\precompute_gesture_noise_levels.py"),
        (Join-Path $projectRoot "code\clone_hand_geometry_noise_caches.py")
    )

    foreach ($requiredPath in $requiredPaths) {
        if (-not (Test-Path $requiredPath)) {
            throw "Required path not found: $requiredPath"
        }
    }

    Write-Host "Project root         : $projectRoot"
    Write-Host "Python               : $python"
    Write-Host "Processed features   : $baseFeatures"
    Write-Host "Output root          : $outputRoot"
    Write-Host "Log directory        : $logDir"
    Write-Host "Run mode             : $runMode"
    Write-Host "Generalization seeds : $($generalizationSeeds -join ', ')"
    Write-Host "Date-holdout seed    : $dateHoldoutSeed"
    Write-Host "Rebuild raw cache    : $RebuildRawFeatureCache"

    Write-Host ""
    Write-Host "Experiment plan:" -ForegroundColor Yellow
    Write-Host "  Clean models                 : 3 training + 3 independent tests"
    Write-Host "  Raw-noise robustness         : 45 training + 45 independent tests"
    Write-Host "  Raw-missing robustness       : 45 training + 45 independent tests"
    Write-Host "  Generalization               : 21 training runs"
    Write-Host "  Factorized robustness        : 31 evaluations"
    Write-Host "  Factorized generalization    : 7 evaluations (full + no_scene)"
    Write-Host "  Final figures and CSV        : generated at the end"

    # --------------------------------------------------------
    # 4.2 Fresh-run archival
    # --------------------------------------------------------

    if (-not $Resume) {
        $backupRoot = Join-Path $outputRoot ("_archive_all_readme_" + $runId)
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null

        $pathsToArchive = @(
            "outputs\baseline_raw_end_to_end",
            "outputs\improved_raw_end_to_end",
            "outputs\hand_geometry_raw_end_to_end",
            "outputs\raw_noise_experiments",
            "outputs\raw_missing_experiments",
            "outputs\generalization",
            "outputs\factorized_robustness",
            "outputs\factorized_generalization",
            "outputs\factorized_head_fusion",
            "outputs\baseline_real_scene_perceiver_io_mptasks",
            "outputs\improved_real_scene_anchor2_perceiver_io_mptasks",
            "outputs\feature_suite\hand_geometry\main",
            "outputs\noise_experiments",
            "outputs\missing_experiments",
            "outputs\factorized_full_summary.csv",
            "outputs\main_results_summary.csv",
            "logs_factorized_full"
        )

        if ($RebuildRawFeatureCache) {
            $pathsToArchive += "outputs\raw_feature_cache"
        }

        foreach ($relativePath in $pathsToArchive) {
            Move-ExistingToBackup `
                -RelativePath $relativePath `
                -BackupRoot $backupRoot
        }

        Write-Host "Fresh-run archive: $backupRoot" -ForegroundColor Yellow
    }

    Clear-ExperimentEnvironment

    # --------------------------------------------------------
    # 4.3 UTF-8 and Python environment preflight
    # --------------------------------------------------------

    Invoke-Step -Name "environment_preflight" -Arguments @(
        "-c",
        "import locale,sys; print('sys.executable:',sys.executable); print('stdout:',sys.stdout.encoding); print('locale:',locale.getpreferredencoding(False)); print('中文测试：README 全量实验开始')"
    )

    # --------------------------------------------------------
    # 4.4 Clean raw-data end-to-end training and testing
    #     Main-run seeds retain the README choices.
    # --------------------------------------------------------

    Invoke-StepMaybe `
        -Name "baseline_clean_train" `
        -RequiredPaths @(
            "outputs\baseline_raw_end_to_end\metrics.json",
            "outputs\baseline_raw_end_to_end\baseline_real_scene_perceiver_io.pt",
            "outputs\baseline_raw_end_to_end\scalers.pkl",
            "outputs\baseline_raw_end_to_end\label_encoder.pkl"
        ) `
        -Arguments @(
            "code\train.py",
            "--model", "baseline",
            "--input-mode", "raw",
            "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
            "--seed", "42",
            "--noise-seed", "42",
            "--epochs", "100",
            "--patience", "10",
            "--output-dir", "outputs\baseline_raw_end_to_end"
        )

    Invoke-StepMaybe `
        -Name "baseline_clean_test" `
        -RequiredPaths @(
            "outputs\baseline_raw_end_to_end\independent_test_metrics.json",
            "outputs\baseline_raw_end_to_end\independent_test_predictions.json"
        ) `
        -Arguments @(
            "code\test.py",
            "--model", "baseline",
            "--input-mode", "raw",
            "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
            "--output-dir", "outputs\baseline_raw_end_to_end"
        )

    Invoke-StepMaybe `
        -Name "improved_clean_train" `
        -RequiredPaths @(
            "outputs\improved_raw_end_to_end\metrics.json",
            "outputs\improved_raw_end_to_end\improved_real_scene_anchor2.pt",
            "outputs\improved_raw_end_to_end\scalers.pkl",
            "outputs\improved_raw_end_to_end\label_encoder.pkl"
        ) `
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

    Invoke-StepMaybe `
        -Name "improved_clean_test" `
        -RequiredPaths @(
            "outputs\improved_raw_end_to_end\independent_test_metrics.json",
            "outputs\improved_raw_end_to_end\independent_test_predictions.json"
        ) `
        -Arguments @(
            "code\test.py",
            "--model", "improved",
            "--input-mode", "raw",
            "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42",
            "--output-dir", "outputs\improved_raw_end_to_end"
        )

    Invoke-StepMaybe `
        -Name "hand_geometry_clean_train" `
        -RequiredPaths @(
            "outputs\hand_geometry_raw_end_to_end\metrics.json",
            "outputs\hand_geometry_raw_end_to_end\improved_real_scene_anchor2.pt",
            "outputs\hand_geometry_raw_end_to_end\scalers.pkl",
            "outputs\hand_geometry_raw_end_to_end\label_encoder.pkl"
        ) `
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

    Invoke-StepMaybe `
        -Name "hand_geometry_clean_test" `
        -RequiredPaths @(
            "outputs\hand_geometry_raw_end_to_end\independent_test_metrics.json",
            "outputs\hand_geometry_raw_end_to_end\independent_test_predictions.json"
        ) `
        -Arguments @(
            "code\test.py",
            "--model", "improved",
            "--input-mode", "raw",
            "--gesture-representation", "hand_geometry",
            "--raw-cache-dir", "outputs\raw_feature_cache\clean_seed42_hand_geometry",
            "--base-feature-dir", $baseFeatures,
            "--output-dir", "outputs\hand_geometry_raw_end_to_end"
        )

    # --------------------------------------------------------
    # 4.5 Gesture raw-noise cache preparation
    # --------------------------------------------------------

    Invoke-StepMaybe `
        -Name "precompute_clip_gesture_noise_levels" `
        -RequiredPaths @(
            "outputs\raw_feature_cache\gesture_noise_20_seed42\gesture_features",
            "outputs\raw_feature_cache\gesture_noise_40_seed42\gesture_features",
            "outputs\raw_feature_cache\gesture_noise_60_seed42\gesture_features"
        ) `
        -Arguments @(
            "code\precompute_gesture_noise_levels.py",
            "--representation", "clip",
            "--levels", "0.2", "0.4", "0.6",
            "--noise-seed", "42",
            "--base-feature-dir", $baseFeatures,
            "--clean-scene-cache", "outputs\raw_feature_cache\clean_seed42\scene_features"
        )

    Invoke-StepMaybe `
        -Name "precompute_hand_geometry_noise_levels" `
        -RequiredPaths @(
            "outputs\raw_feature_cache\gesture_noise_20_seed42_hand_geometry\gesture_features",
            "outputs\raw_feature_cache\gesture_noise_40_seed42_hand_geometry\gesture_features",
            "outputs\raw_feature_cache\gesture_noise_60_seed42_hand_geometry\gesture_features"
        ) `
        -Arguments @(
            "code\precompute_gesture_noise_levels.py",
            "--representation", "hand_geometry",
            "--levels", "0.2", "0.4", "0.6",
            "--noise-seed", "42",
            "--base-feature-dir", $baseFeatures,
            "--clean-scene-cache", "outputs\raw_feature_cache\clean_seed42\scene_features"
        )
    # --------------------------------------------------------
    # 4.6 Raw-noise robustness training
    # --------------------------------------------------------

    $baselineNoiseArgs = @(
        "code\run_noise_experiments.py",
        "--model", "baseline",
        "--output-model-name", "baseline",
        "--input-mode", "raw",
        "--noise-space", "raw",
        "--base-feature-dir", $baseFeatures,
        "--noise-seed", "42",
        "--epochs", "100",
        "--patience", "10",
        "--execute"
    )

    if ($Resume) {
        $baselineNoiseArgs += "--skip-existing"
    }

    $env:SMART_AR_RANDOM_SEED = "42"
    Invoke-Step -Name "baseline_raw_noise_train" -Arguments $baselineNoiseArgs

    # Baseline raw-noise preprocessing creates the IMU/Audio/Text/Scene
    # perturbation caches. Reuse those perturbations for Hand Geometry only
    # after the Baseline noise suite has completed.
    Invoke-Step -Name "clone_hand_geometry_noise_caches" -Arguments @(
        "code\clone_hand_geometry_noise_caches.py",
        "--levels", "0.2", "0.4", "0.6",
        "--modalities", "imu", "audio", "text", "scene",
        "--noise-seed", "42",
        "--clean-feature-dir", $baseFeatures
    )

    $improvedNoiseArgs = @(
        "code\run_noise_experiments.py",
        "--model", "improved",
        "--output-model-name", "improved",
        "--input-mode", "raw",
        "--noise-space", "raw",
        "--base-feature-dir", $baseFeatures,
        "--noise-seed", "42",
        "--epochs", "100",
        "--patience", "4",
        "--execute"
    )

    if ($Resume) {
        $improvedNoiseArgs += "--skip-existing"
    }

    $env:SMART_AR_RANDOM_SEED = "42"
    Invoke-Step -Name "improved_raw_noise_train" -Arguments $improvedNoiseArgs

    $handNoiseArgs = @(
        "code\run_noise_experiments.py",
        "--model", "improved",
        "--output-model-name", "hand_geometry",
        "--input-mode", "raw",
        "--noise-space", "raw",
        "--gesture-representation", "hand_geometry",
        "--base-feature-dir", $baseFeatures,
        "--noise-seed", "42",
        "--epochs", "100",
        "--patience", "4",
        "--execute"
    )

    if ($Resume) {
        $handNoiseArgs += "--skip-existing"
    }

    $env:SMART_AR_RANDOM_SEED = "42"
    Invoke-Step -Name "hand_geometry_raw_noise_train" -Arguments $handNoiseArgs

    # --------------------------------------------------------
    # 4.7 Independent tests for all 45 raw-noise checkpoints
    # --------------------------------------------------------

    $noiseConfigs = @(
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

    foreach ($config in $noiseConfigs) {
        foreach ($modality in $modalities) {
            foreach ($percent in $percentages) {
                $level = ($percent / 100.0).ToString(
                    [System.Globalization.CultureInfo]::InvariantCulture
                )

                $suffix = if ($config.Representation -eq "hand_geometry") {
                    "_hand_geometry"
                }
                else {
                    ""
                }

                $cacheDir = "outputs\raw_feature_cache\${modality}_noise_${percent}_seed42${suffix}"
                $outputDir = "outputs\raw_noise_experiments\$($config.Name)\${modality}_noise_$percent"
                $testName = "$($config.Name)_${modality}_noise_${percent}_test"

                Invoke-StepMaybe `
                    -Name $testName `
                    -RequiredPaths @(
                        "$outputDir\independent_test_metrics.json",
                        "$outputDir\independent_test_predictions.json"
                    ) `
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

    # --------------------------------------------------------
    # 4.8 Raw-missing robustness:
    #     5 single-modality + 10 double-modality conditions
    # --------------------------------------------------------

    Clear-ExperimentEnvironment

    $baselineMissingArgs = @(
        "code\run_missing_experiments.py",
        "--model", "baseline",
        "--output-model-name", "baseline",
        "--input-mode", "raw",
        "--gesture-representation", "clip",
        "--base-feature-dir", $cleanClip,
        "--seed", "42",
        "--noise-seed", "42",
        "--epochs", "100",
        "--patience", "10",
        "--max-missing", "2",
        "--execute"
    )

    if ($Resume) {
        $baselineMissingArgs += "--skip-existing"
    }

    Invoke-Step -Name "baseline_raw_missing" -Arguments $baselineMissingArgs

    $improvedMissingArgs = @(
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
        "--execute"
    )

    if ($Resume) {
        $improvedMissingArgs += "--skip-existing"
    }

    Invoke-Step -Name "improved_raw_missing" -Arguments $improvedMissingArgs

    $handMissingArgs = @(
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
        "--execute"
    )

    if ($Resume) {
        $handMissingArgs += "--skip-existing"
    }

    Invoke-Step -Name "hand_geometry_raw_missing" -Arguments $handMissingArgs

    # --------------------------------------------------------
    # 4.9 Generalization training
    #     Every model uses seeds 7, 42 and 123.
    #     The four date holdouts retain README seed 42.
    # --------------------------------------------------------

    Clear-ExperimentEnvironment

    $baselineGeneralizationArgs = @(
        "code\run_generalization_experiments.py",
        "--model", "baseline",
        "--output-model-name", "baseline",
        "--seeds", $generalizationSeeds[0], $generalizationSeeds[1], $generalizationSeeds[2],
        "--epochs", "100",
        "--patience", "10",
        "--execute"
    )

    if ($Resume) {
        $baselineGeneralizationArgs += "--skip-existing"
    }

    Invoke-Step `
        -Name "baseline_generalization_all" `
        -Arguments $baselineGeneralizationArgs

    $improvedGeneralizationArgs = @(
        "code\run_generalization_experiments.py",
        "--model", "improved",
        "--output-model-name", "improved",
        "--seeds", $generalizationSeeds[0], $generalizationSeeds[1], $generalizationSeeds[2],
        "--epochs", "100",
        "--patience", "4",
        "--execute"
    )

    if ($Resume) {
        $improvedGeneralizationArgs += "--skip-existing"
    }

    Invoke-Step `
        -Name "improved_generalization_all" `
        -Arguments $improvedGeneralizationArgs

    $handGeneralizationArgs = @(
        "code\run_generalization_experiments.py",
        "--model", "improved",
        "--output-model-name", "hand_geometry",
        "--gesture-feature-dir", $handGeometryFeatureDir,
        "--gesture-feature-dim", "96",
        "--seeds", $generalizationSeeds[0], $generalizationSeeds[1], $generalizationSeeds[2],
        "--epochs", "100",
        "--patience", "4",
        "--execute"
    )

    if ($Resume) {
        $handGeneralizationArgs += "--skip-existing"
    }

    Invoke-Step `
        -Name "hand_geometry_generalization_all" `
        -Arguments $handGeneralizationArgs

    # --------------------------------------------------------
    # 4.10 Publish clean-output compatibility directories
    #      required by the current Factorized/visualization scripts.
    # --------------------------------------------------------

    Sync-Directory `
        -Source (Join-Path $outputRoot "baseline_raw_end_to_end") `
        -Destination (Join-Path $outputRoot "baseline_real_scene_perceiver_io_mptasks")

    Sync-Directory `
        -Source (Join-Path $outputRoot "improved_raw_end_to_end") `
        -Destination (Join-Path $outputRoot "improved_real_scene_anchor2_perceiver_io_mptasks")

    Sync-Directory `
        -Source (Join-Path $outputRoot "hand_geometry_raw_end_to_end") `
        -Destination (Join-Path $outputRoot "feature_suite\hand_geometry\main")

    # --------------------------------------------------------
    # 4.11 Factorized Heads
    #      - full clean result
    #      - 15 missing scenarios
    #      - 15 noise scenarios
    #      - 3 repeated seeds
    #      - 4 date holdouts
    #      - full and no_scene generalization evaluation
    #
    # Improved and Hand Geometry generalization models were trained
    # in the previous stage, so duplicate training is disabled here.
    # --------------------------------------------------------

    $factorizedRequiredPaths = @(
    "outputs\factorized_robustness\full\factorized_head_fusion.json",

    "outputs\factorized_generalization\seed7_default_test\factorized_head_fusion.json",
    "outputs\factorized_generalization\seed42_default_test\factorized_head_fusion.json",
    "outputs\factorized_generalization\seed123_default_test\factorized_head_fusion.json",

    "outputs\factorized_generalization\date_20260131_test_seed42\factorized_head_fusion.json",
    "outputs\factorized_generalization\date_20260227_test_seed42\factorized_head_fusion.json",
    "outputs\factorized_generalization\date_20260301_test_seed42\factorized_head_fusion.json",
    "outputs\factorized_generalization\date_20260306_test_seed42\factorized_head_fusion.json"
)

$factorizedModalities = @(
    "imu",
    "gesture",
    "audio",
    "text",
    "scene"
)

foreach ($modality in $factorizedModalities) {
    $factorizedRequiredPaths += `
        "outputs\factorized_robustness\no_$modality\factorized_head_fusion.json"

    foreach ($level in @(20, 40, 60)) {
        $factorizedRequiredPaths += `
            "outputs\factorized_robustness\${modality}_noise_$level\factorized_head_fusion.json"
    }
}

for ($i = 0; $i -lt $factorizedModalities.Count; $i++) {
    for ($j = $i + 1; $j -lt $factorizedModalities.Count; $j++) {
        $first = $factorizedModalities[$i]
        $second = $factorizedModalities[$j]

        $factorizedRequiredPaths += `
            "outputs\factorized_robustness\no_${first}_${second}\factorized_head_fusion.json"
    }
}

Invoke-StepMaybe `
    -Name "factorized_full_suite" `
    -RequiredPaths $factorizedRequiredPaths `
    -Arguments @(
        "code\run_factorized_full_suite.py",
        "--epochs", "100",
        "--patience", "4",
        "--seeds",
        $generalizationSeeds[0],
        $generalizationSeeds[1],
        $generalizationSeeds[2],
        "--geometry-feature-dir", $handGeometryFeatureDir,
        "--scene-gesture-feature-dir", $sceneGestureFeatureDir,
        "--skip-generalization-training",
        "--execute"
    )

    # Current visualization code expects the clean factorized result
    # at outputs\factorized_head_fusion\factorized_head_fusion.json.
    $factorizedFullSource = Join-Path `
        $outputRoot `
        "factorized_robustness\full\factorized_head_fusion.json"

    $factorizedMainDir = Join-Path $outputRoot "factorized_head_fusion"
    $factorizedMainDestination = Join-Path `
        $factorizedMainDir `
        "factorized_head_fusion.json"

    if (-not (Test-Path $factorizedFullSource)) {
        throw "Factorized full result not found: $factorizedFullSource"
    }

    New-Item -ItemType Directory -Force -Path $factorizedMainDir | Out-Null
    Copy-Item `
        -LiteralPath $factorizedFullSource `
        -Destination $factorizedMainDestination `
        -Force

    # The current visualization script reads legacy robustness directory
    # names. Publish only lightweight metrics aliases from the compliant
    # raw-data experiments; checkpoints are not duplicated.
    Publish-MetricsTree `
        -SourceRoot (Join-Path $outputRoot "raw_noise_experiments") `
        -DestinationRoot (Join-Path $outputRoot "noise_experiments")

    Publish-MetricsTree `
        -SourceRoot (Join-Path $outputRoot "raw_missing_experiments") `
        -DestinationRoot (Join-Path $outputRoot "missing_experiments")

    # --------------------------------------------------------
    # 4.12 Figures and summary CSV files
    # --------------------------------------------------------

    Invoke-Step -Name "visualize_wjy_results" -Arguments @(
        "code\visualize_wjy_results.py"
    )

    # --------------------------------------------------------
    # 4.13 Completeness checks
    # --------------------------------------------------------

    Write-Host ""
    Write-Host "===== Final completeness checks =====" -ForegroundColor Cyan

    $cleanOutputs = @(
        "baseline_raw_end_to_end",
        "improved_raw_end_to_end",
        "hand_geometry_raw_end_to_end"
    )

    foreach ($directoryName in $cleanOutputs) {
        $directory = Join-Path $outputRoot $directoryName

        foreach ($requiredName in @(
            "metrics.json",
            "independent_test_metrics.json",
            "independent_test_predictions.json"
        )) {
            $requiredPath = Join-Path $directory $requiredName

            if (-not (Test-Path $requiredPath)) {
                throw "Missing clean-run artifact: $requiredPath"
            }
        }
    }

    foreach ($modelName in @("baseline", "improved", "hand_geometry")) {
        $noiseRoot = Join-Path $outputRoot "raw_noise_experiments\$modelName"
        $noiseTrainCount = Count-Files -Root $noiseRoot -Filter "metrics.json"
        $noiseTestCount = Count-Files -Root $noiseRoot -Filter "independent_test_metrics.json"

        Write-Host "$modelName raw-noise: train=$noiseTrainCount/15 independent_test=$noiseTestCount/15"

        if ($noiseTrainCount -ne 15 -or $noiseTestCount -ne 15) {
            throw "$modelName raw-noise results are incomplete"
        }

        $missingRoot = Join-Path $outputRoot "raw_missing_experiments\$modelName"
        $missingTrainCount = Count-Files -Root $missingRoot -Filter "metrics.json"
        $missingTestCount = Count-Files -Root $missingRoot -Filter "independent_test_metrics.json"

        Write-Host "$modelName raw-missing: train=$missingTrainCount/15 independent_test=$missingTestCount/15"

        if ($missingTrainCount -ne 15 -or $missingTestCount -ne 15) {
            throw "$modelName raw-missing results are incomplete"
        }

        $generalizationRoot = Join-Path $outputRoot "generalization"
        $generalizationCount = @(
            Get-ChildItem `
                $generalizationRoot `
                -Directory `
                -Filter "${modelName}_*" |
                Where-Object {
                    Test-Path (Join-Path $_.FullName "metrics.json")
                }
        ).Count

        Write-Host "$modelName generalization: $generalizationCount/7"

        if ($generalizationCount -ne 7) {
            throw "$modelName generalization results are incomplete: $generalizationCount/7"
        }
    }

    $factorizedRobustnessCount = Count-Files `
        -Root (Join-Path $outputRoot "factorized_robustness") `
        -Filter "factorized_head_fusion.json"

    $factorizedGeneralizationCount = Count-Files `
        -Root (Join-Path $outputRoot "factorized_generalization") `
        -Filter "factorized_head_fusion.json"

    Write-Host "Factorized robustness: $factorizedRobustnessCount/31"
    Write-Host "Factorized generalization: $factorizedGeneralizationCount/7"

    if ($factorizedRobustnessCount -ne 31) {
        throw "Factorized robustness results are incomplete: $factorizedRobustnessCount/31"
    }

    if ($factorizedGeneralizationCount -ne 7) {
        throw "Factorized generalization results are incomplete: $factorizedGeneralizationCount/7"
    }

    foreach ($requiredSummary in @(
        "outputs\factorized_full_summary.csv",
        "outputs\factorized_head_fusion\factorized_head_fusion.json"
    )) {
        $requiredSummaryPath = Join-Path $projectRoot $requiredSummary

        if (-not (Test-Path $requiredSummaryPath)) {
            throw "Missing summary artifact: $requiredSummaryPath"
        }
    }

    # --------------------------------------------------------
    # 4.14 Timing-field validation
    # --------------------------------------------------------

    $timingCheckSource = @'
import json
from pathlib import Path

root = Path("outputs")

train_files = [
    root / "baseline_raw_end_to_end" / "metrics.json",
    root / "improved_raw_end_to_end" / "metrics.json",
    root / "hand_geometry_raw_end_to_end" / "metrics.json",
]
train_files += list((root / "raw_noise_experiments").glob("*/*/metrics.json"))
train_files += list((root / "raw_missing_experiments").glob("*/*/metrics.json"))
train_files += list((root / "generalization").glob("*/metrics.json"))

test_files = [
    root / "baseline_raw_end_to_end" / "independent_test_metrics.json",
    root / "improved_raw_end_to_end" / "independent_test_metrics.json",
    root / "hand_geometry_raw_end_to_end" / "independent_test_metrics.json",
]
test_files += list(
    (root / "raw_noise_experiments").glob(
        "*/*/independent_test_metrics.json"
    )
)
test_files += list(
    (root / "raw_missing_experiments").glob(
        "*/*/independent_test_metrics.json"
    )
)

missing_train = []
for path in train_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timing = payload.get("timing", {})
    if "avg_training_seconds_per_sample" not in timing:
        missing_train.append(str(path))

missing_test = []
for path in test_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timing = payload.get("timing", {})
    if "avg_test_seconds_per_sample" not in timing:
        missing_test.append(str(path))

print(f"training timing files: {len(train_files)}")
print(f"testing timing files: {len(test_files)}")
print(f"missing training timing: {len(missing_train)}")
print(f"missing testing timing: {len(missing_test)}")

if missing_train:
    print("Missing avg_training_seconds_per_sample:")
    for path in missing_train:
        print("  ", path)

if missing_test:
    print("Missing avg_test_seconds_per_sample:")
    for path in missing_test:
        print("  ", path)

if missing_train or missing_test:
    raise SystemExit(1)

print("All required per-sample timing fields are present.")
'@

    Invoke-Step -Name "validate_timing_fields" -Arguments @(
        "-c",
        $timingCheckSource
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "ALL README AND COURSE EXPERIMENTS COMPLETED" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Logs: $logDir"
    Write-Host "Generalization seeds: $($generalizationSeeds -join ', ')"
    Write-Host "Date-holdout seed: $dateHoldoutSeed"
}
finally {
    if (Test-Path $helperRoot) {
        Remove-Item -LiteralPath $helperRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
