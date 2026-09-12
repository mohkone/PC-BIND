param(
    [switch]$Force,
    [switch]$SkipEnsemble
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found. Run .\setup_env.ps1 first."
}

function Run-SeedAblation {
    param(
        [int]$Seed,
        [string]$OutputDir,
        [string]$Mode
    )

    $doneCheckpoint = Join-Path $ProjectRoot (Join-Path $OutputDir "fold5_best.pt")
    $doneMetrics = Join-Path $ProjectRoot (Join-Path $OutputDir "metrics_summary.csv")
    if (-not $Force -and (Test-Path $doneCheckpoint) -and (Test-Path $doneMetrics)) {
        Write-Host "Skipping seed $Seed ($Mode): $OutputDir already looks complete."
        return
    }

    Write-Host ""
    Write-Host "=============================="
    Write-Host "Running $Mode ablation seed $Seed -> $OutputDir"
    Write-Host "=============================="

    $commonArgs = @{
        Seed = $Seed
        OutputDir = $OutputDir
        TwoHead = 1
        RankLossWeight = "0.55"
        RankFusion = "0.35"
        RankWarmupEpochs = 3
        ModelDropout = "0.25"
        EdgeDropout = "0.06"
        WeightDecay = "3e-4"
    }

    if ($Mode -eq "no_plm") {
        & (Join-Path $ProjectRoot "run_seed.ps1") @commonArgs -NoPlm 1
    } elseif ($Mode -eq "no_aux") {
        & (Join-Path $ProjectRoot "run_seed.ps1") @commonArgs -NoAuxObjective 1
    } else {
        throw "Unknown mode: $Mode"
    }
}

function Run-EnsembleIfReady {
    param(
        [string]$Name,
        [string[]]$Runs,
        [string]$OutputDir
    )

    if ($SkipEnsemble) {
        Write-Host "Skipping $Name ensemble evaluation because -SkipEnsemble was set."
        return
    }

    $rankingPath = Join-Path $ProjectRoot (Join-Path $OutputDir "strategy_ranking.csv")
    if (-not $Force -and (Test-Path $rankingPath)) {
        Write-Host "Skipping $Name ensemble evaluation: $OutputDir already has strategy_ranking.csv."
        return
    }

    foreach ($runSpec in $Runs) {
        $runDir = ($runSpec -split ":", 2)[0]
        $ckpt = Join-Path $ProjectRoot (Join-Path $runDir "fold5_best.pt")
        if (-not (Test-Path $ckpt)) {
            throw "Cannot evaluate $Name ensemble; missing $ckpt"
        }
    }

    Write-Host ""
    Write-Host "=============================="
    Write-Host "Evaluating $Name multi-seed ensemble -> $OutputDir"
    Write-Host "=============================="
    & $VenvPython (Join-Path $ProjectRoot "evaluate_multi_seed_ensembles.py") --runs @Runs --output-dir $OutputDir
}

$noPlmRuns = @(
    @{ Seed = 2041; OutputDir = "outputs_no_plm_seed2041" },
    @{ Seed = 2042; OutputDir = "outputs_no_plm_seed2042" },
    @{ Seed = 2043; OutputDir = "outputs_no_plm_seed2043" }
)

$noAuxRuns = @(
    @{ Seed = 2051; OutputDir = "outputs_no_aux_seed2051" },
    @{ Seed = 2052; OutputDir = "outputs_no_aux_seed2052" },
    @{ Seed = 2053; OutputDir = "outputs_no_aux_seed2053" }
)

foreach ($run in $noPlmRuns) {
    Run-SeedAblation -Seed $run.Seed -OutputDir $run.OutputDir -Mode "no_plm"
}

foreach ($run in $noAuxRuns) {
    Run-SeedAblation -Seed $run.Seed -OutputDir $run.OutputDir -Mode "no_aux"
}

Run-EnsembleIfReady `
    -Name "no_plm" `
    -Runs @("outputs_no_plm_seed2041:2041", "outputs_no_plm_seed2042:2042", "outputs_no_plm_seed2043:2043") `
    -OutputDir "outputs_multiseed_no_plm_2041_2042_2043_ckptfusion"

Run-EnsembleIfReady `
    -Name "no_aux" `
    -Runs @("outputs_no_aux_seed2051:2051", "outputs_no_aux_seed2052:2052", "outputs_no_aux_seed2053:2053") `
    -OutputDir "outputs_multiseed_no_aux_2051_2052_2053_ckptfusion"

Write-Host ""
Write-Host "Reviewer ablation runs complete."
Write-Host "Next: .\.venv\Scripts\python.exe .\summarize_reviewer_ablations.py"
