param(
    [int]$Seed = 2101,
    [string]$OutputDir = "",
    [int]$BatchSize = 1
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_pcbind_ot_ablation_seed$Seed"
}

$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl,TestUB25.pkl"

# Ablation: same sparse pair-affinity MLP and dustbin, without transport
# column-cap iterations. This isolates the cross-target competition.
$env:PPI_PARTNER_TRANSPORT = "1"
$env:PPI_TRANSPORT_SINKHORN_ITERS = "0"
$env:PPI_TRANSPORT_TAU = "1.0"
$env:PPI_TRANSPORT_DUSTBIN = "1"
$env:PPI_TRANSPORT_TOP_K = "32"
$env:PPI_TRANSPORT_ENTROPY_WEIGHT = "0.0"
$env:PPI_TRANSPORT_SPARSITY_WEIGHT = "0.0"
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
