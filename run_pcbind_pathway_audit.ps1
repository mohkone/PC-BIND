param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$OutputDir = "",
    [int]$BootstrapReplicates = 500,
    [int]$BatchSize = 4,
    [int]$MismatchSeed = 12017,
    [int]$ShuffleSeed = 13017,
    [int]$NeighborCount = 10
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $leaf = Split-Path -Leaf $RunDir
    $OutputDir = "${leaf}_pathway_audit"
}

Push-Location $ProjectRoot
try {
    & $VenvPython (Join-Path $ProjectRoot "evaluate_pcbind_pathway_utilization.py") `
        --run-dir $RunDir `
        --output-dir $OutputDir `
        --bootstrap-replicates $BootstrapReplicates `
        --batch-size $BatchSize `
        --mismatch-seed $MismatchSeed `
        --shuffle-seed $ShuffleSeed `
        --neighbor-count $NeighborCount
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
