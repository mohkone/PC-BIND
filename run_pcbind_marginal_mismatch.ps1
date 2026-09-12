param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$MarginalDir = "",
    [string]$OutputDir = "",
    [int]$MismatchSeed = 9917,
    [int]$NeighborCount = 10,
    [int]$BatchSize = 4,
    [string]$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}

if ([string]::IsNullOrWhiteSpace($MarginalDir)) {
    $MarginalDir = "${RunDir}_marginal_raw"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "${RunDir}_marginal_mismatch_control"
}

$MarginalSummary = Join-Path $MarginalDir "pcbind_marginal_summary.json"
if (-not (Test-Path -LiteralPath $MarginalSummary)) {
    throw "Missing raw marginal summary: $MarginalSummary"
}

& $VenvPython `
    (Join-Path $ProjectRoot "evaluate_pcbind_marginal_mismatch.py") `
    --run-dir $RunDir `
    --marginal-summary $MarginalSummary `
    --output-dir $OutputDir `
    --mismatch-seed $MismatchSeed `
    --neighbor-count $NeighborCount `
    --batch-size $BatchSize `
    --test-sets $TestSets

exit $LASTEXITCODE
