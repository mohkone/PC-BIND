param(
    [string]$RunDir = "outputs_pcbind_v5_pair_seed2101",
    [string]$OutputDir = "outputs_pcbind_v5_pair_seed2101_marginal_raw",
    [int]$BatchSize = 1,
    [string]$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Evaluator = Join-Path $ProjectRoot "evaluate_pcbind_marginal.py"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}

Push-Location $ProjectRoot
try {
    & $VenvPython $Evaluator `
        --run-dir $RunDir `
        --output-dir $OutputDir `
        --batch-size $BatchSize `
        --test-sets $TestSets `
        --no-smooth-marginal
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $SummaryPath = Join-Path $OutputDir "pcbind_marginal_summary.json"
    if (-not (Test-Path $SummaryPath)) {
        throw "Raw marginal summary was not created: $SummaryPath"
    }
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    if ($Summary.smooth_marginal -ne $false) {
        throw "Raw marginal guard failed: smooth_marginal is not false in $SummaryPath"
    }
    Write-Host "Verified raw marginal evaluation: smooth_marginal=False"
}
finally {
    Pop-Location
}
