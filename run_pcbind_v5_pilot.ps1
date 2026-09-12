param(
    [int]$Seed = 2101,
    [string]$OutputDir = "",
    [int]$BatchSize = 1,
    [int]$SmokeOnly = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_pcbind_v5_shared_seed$Seed"
}

$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl,TestUB25.pkl"
$FullPartnerDataFiles = @(
    "Train335.pkl",
    "Test60.pkl",
    "Test287.pkl",
    "TestB25.pkl"
)

Push-Location $ProjectRoot
try {
    & $VenvPython `
        (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
        --require-plm `
        --require-partner `
        --require-pair `
        --require-partner-encoder `
        @FullPartnerDataFiles
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # TestUB25 contains 12 valid partner inputs and 13 documented target-pathway
    # fallbacks. Validate that the frozen coverage has not fallen below 12/25.
    & $VenvPython `
        (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
        --require-plm `
        --require-partner `
        --require-pair `
        --require-partner-encoder `
        --min-partner-encoder-coverage 0.48 `
        "TestUB25.pkl"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython (Join-Path $ProjectRoot "smoke_test_pcbind_v5.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

if ($SmokeOnly -ne 0) {
    Write-Host "PC-BIND v5 smoke-only run complete."
    exit 0
}

& (Join-Path $ProjectRoot "run_seed.ps1") `
    -Seed $Seed `
    -OutputDir $OutputDir `
    -GroupedCV 1 `
    -CVGroupKey "complex_code" `
    -TestSets $TestSets `
    -TwoHead 1 `
    -RankLossWeight "0.55" `
    -RankFusion "0.35" `
    -RankWarmupEpochs 3 `
    -ModelDropout "0.25" `
    -EdgeDropout "0.06" `
    -WeightDecay "3e-4" `
    -PartnerConditioning 1 `
    -PartnerTopK 16 `
    -PartnerDirectFusion 1 `
    -PartnerLogitMode "standard" `
    -PartnerResidueEncoder 1 `
    -PartnerEncoderLayers 1 `
    -PartnerTargetFusion "0.25" `
    -PartnerContrast 0 `
    -PairContactLoss 0 `
    -PairMarginalConsistency 0 `
    -PairMarginalContrast 0 `
    -PairContactContrast 0 `
    -BatchSize $BatchSize `
    -UsePlm 1 `
    -UseAuxPlm 1 `
    -PartnerContactAux 1 `
    -PatchLabels 0 `
    -SelectionMetric "mcc" `
    -SelectionAuprWeight "0.35"
