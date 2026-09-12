param(
    [int]$Seed = 2101,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_grouped_full_seed$Seed"
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}

Push-Location $ProjectRoot
try {
    & $VenvPython `
        (Join-Path $ProjectRoot "audit_pcbind_pairing.py") `
        --seed $Seed `
        "Train335.pkl" `
        "Test60.pkl" `
        "Test287.pkl" `
        "TestB25.pkl" `
        "TestUB25.pkl"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

& (Join-Path $ProjectRoot "run_seed.ps1") `
    -Seed $Seed `
    -OutputDir $OutputDir `
    -GroupedCV 1 `
    -CVGroupKey "complex_code" `
    -TestSets "Test60.pkl,Test287.pkl,TestB25.pkl,TestUB25.pkl" `
    -TwoHead 1 `
    -RankLossWeight "0.55" `
    -RankFusion "0.35" `
    -RankWarmupEpochs 3 `
    -ModelDropout "0.25" `
    -EdgeDropout "0.06" `
    -WeightDecay "3e-4" `
    -PartnerConditioning 0 `
    -PartnerContrast 0 `
    -PairContactLoss 0 `
    -PairMarginalConsistency 0 `
    -PairMarginalContrast 0 `
    -PairContactContrast 0 `
    -BatchSize 1 `
    -UsePlm 1 `
    -UseAuxPlm 1 `
    -PartnerContactAux 1 `
    -PatchLabels 0 `
    -SelectionMetric "mcc" `
    -SelectionAuprWeight "0.35"
