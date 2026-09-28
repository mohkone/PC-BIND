param(
    [int]$Seed = 2101,
    [string]$OutputDir = "",
    [int]$BatchSize = 1,
    [int]$SmokeOnly = 0,
    [int]$PartnerTransport = 1,
    [double]$TransportTau = 0.1,
    [int]$TransportSinkhornIters = 5,
    [double]$TransportLogitFusion = 0.3,
    [ValidateRange(1, 5)][int]$MaxFolds = 5,
    [int]$SkipTestEval = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_pcbind_ot_seed$Seed"
}

$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl,TestUB25.pkl"
$FullPartnerDataFiles = @(
    "Train335.pkl",
    "Test60.pkl",
    "Test287.pkl",
    "TestB25.pkl"
)

if (-not $DryRun) {
Push-Location $ProjectRoot
try {
    & $VenvPython `
        (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
        --require-plm `
        --require-partner `
        --require-pair `
        --require-partner-encoder `
        @FullPartnerDataFiles
    if ($LASTEXITCODE -ne 0) { throw "Prerequisite check failed: $LASTEXITCODE" }

    & $VenvPython `
        (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
        --require-plm `
        --require-partner `
        --require-pair `
        --require-partner-encoder `
        --min-partner-encoder-coverage 0.48 `
        "TestUB25.pkl"
    if ($LASTEXITCODE -ne 0) { throw "Prerequisite check failed: $LASTEXITCODE" }

    & $VenvPython (Join-Path $ProjectRoot "smoke_test_pcbind_v5.py")
    if ($LASTEXITCODE -ne 0) { throw "Prerequisite check failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
}

# The transport branch is a capacity-constrained, OT-inspired matching model.
# A data-free branch smoke test must also run for -SmokeOnly.
if (-not $DryRun -and $PartnerTransport -ne 0) {
    Push-Location $ProjectRoot
    try {
        & $VenvPython (Join-Path $ProjectRoot "smoke_test_pcbind_ot.py")
        if ($LASTEXITCODE -ne 0) { throw "Transport smoke test failed: $LASTEXITCODE" }
    } finally { Pop-Location }
}
if ($SmokeOnly -ne 0) {
    Write-Host "PC-BIND matching-path smoke-only run complete."
    return
}

& (Join-Path $ProjectRoot "run_seed.ps1") `
    -Seed $Seed `
    -PartnerTransport $PartnerTransport `
    -TransportSinkhornIters $TransportSinkhornIters `
    -TransportTau $TransportTau.ToString([System.Globalization.CultureInfo]::InvariantCulture) `
    -TransportLogitFusion $TransportLogitFusion.ToString([System.Globalization.CultureInfo]::InvariantCulture) `
    -SkipTestEval $SkipTestEval `
    -DryRun:$DryRun `
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
    -PartnerDirectFusion 0 `
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
    -MaxFolds $MaxFolds `
    -UsePlm 1 `
    -UseAuxPlm 1 `
    -PartnerContactAux 1 `
    -PatchLabels 0 `
    -SelectionMetric "mcc" `
    -SelectionAuprWeight "0.35"
