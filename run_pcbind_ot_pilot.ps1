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
    $OutputDir = "outputs_pcbind_ot_seed$Seed"
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
    Write-Host "PC-BIND-OT smoke-only run complete."
    exit 0
}

# PC-BIND-OT: sparse unbalanced optimal transport partner matching
# Key differences from v5:
#   - PartnerTransport 1: enables the SparseTransportLayer
#   - TransportSinkhornIters 5: log-domain Sinkhorn iterations
#   - TransportTau 0.1: temperature controlling plan sharpness
#   - TransportDustbin 1: per-residue "no match" dustbin column
#   - TransportTopK 32: wider candidate set than v5's 16
#   - TransportEntropyWeight 0.01: entropy regularisation for sharp plans
#   - TransportSparsityWeight 0.005: concentration regularisation
#   - TransportLogitFusion 0.3: weight of transport-derived logits
$env:PPI_PARTNER_TRANSPORT = "1"
$env:PPI_TRANSPORT_SINKHORN_ITERS = "5"
$env:PPI_TRANSPORT_TAU = "0.1"
$env:PPI_TRANSPORT_DUSTBIN = "1"
$env:PPI_TRANSPORT_TOP_K = "32"
$env:PPI_TRANSPORT_ENTROPY_WEIGHT = "0.01"
$env:PPI_TRANSPORT_SPARSITY_WEIGHT = "0.005"
$env:PPI_TRANSPORT_LOGIT_FUSION = "0.3"

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
    -UsePlm 1 `
    -UseAuxPlm 1 `
    -PartnerContactAux 1 `
    -PatchLabels 0 `
    -SelectionMetric "mcc" `
    -SelectionAuprWeight "0.35"
