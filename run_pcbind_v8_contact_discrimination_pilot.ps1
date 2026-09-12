param(
    [int]$Seed = 2111,
    [string]$OutputDir = "",
    [int]$BatchSize = 4,
    [int]$MaxFolds = 1,
    [int]$SkipTestEval = 1,
    [int]$ContactContrast = 1,
    [string]$PairContactLossWeight = "0.005",
    [string]$MarginalConsistencyWeight = "0.10",
    [string]$ContactContrastWeight = "0.005",
    [string]$ContactContrastMargin = "0.03",
    [int]$HardK = 10,
    [int]$WarmupEpochs = 5,
    [int]$SmokeOnly = 0
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ($ContactContrast -ne 0 -and $BatchSize -lt 2) {
    throw "Contact-level partner discrimination requires BatchSize >= 2."
}
if ($HardK -lt 1 -or $HardK -gt 16) {
    throw "HardK must be between 1 and the fixed sparse candidate count (16)."
}
if ($MaxFolds -lt 1 -or $MaxFolds -gt 5) {
    throw "MaxFolds must be between 1 and 5."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $variant = if ($ContactContrast -ne 0) { "contrast" } else { "control" }
    $OutputDir = "outputs_pcbind_v8_contact_${variant}_seed$Seed"
}

$TestSets = "Test60.pkl,Test287.pkl,TestB25.pkl"
$DataFiles = @(
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
        @DataFiles
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython (Join-Path $ProjectRoot "smoke_test_pcbind_v5.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

if ($SmokeOnly -ne 0) {
    Write-Host "PC-BIND v8 contact-discrimination smoke-only run complete."
    exit 0
}

& (Join-Path $ProjectRoot "run_seed.ps1") `
    -Seed $Seed `
    -OutputDir $OutputDir `
    -GroupedCV 1 `
    -CVGroupKey "complex_code" `
    -TestSets $TestSets `
    -MaxFolds $MaxFolds `
    -SkipTestEval $SkipTestEval `
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
    -PairContactLoss 1 `
    -PairContactLossWeight $PairContactLossWeight `
    -PairContactPosWeight "8.0" `
    -PairContactHead "mlp" `
    -PairContactWarmupEpochs $WarmupEpochs `
    -PairMarginalConsistency 1 `
    -PairMarginalConsistencyWeight $MarginalConsistencyWeight `
    -PairMarginalConsistencyWarmupEpochs $WarmupEpochs `
    -PairMarginalContrast 0 `
    -PairContactContrast $ContactContrast `
    -PairContactContrastWeight $ContactContrastWeight `
    -PairContactContrastMargin $ContactContrastMargin `
    -PairContactContrastHardK $HardK `
    -PairContactContrastWarmupEpochs $WarmupEpochs `
    -BatchSize $BatchSize
