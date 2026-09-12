param(
    [int]$Seed = 2111,
    [int]$Residual = 1,
    [string]$Alpha = "0.25",
    [string]$OutputDir = "",
    [int]$BatchSize = 4,
    [string]$PairContactLossWeight = "0.005",
    [int]$SmokeOnly = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ($Residual -ne 0 -and $Residual -ne 1) {
    throw "Residual must be 0 (matched alpha=0 control) or 1 (v9 treatment)."
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be positive."
}
$AlphaValue = if ($Residual -eq 0) { "0.0" } else { $Alpha }
if ([double]$AlphaValue -lt 0.0 -or [double]$AlphaValue -gt 1.0) {
    throw "Alpha must be between 0 and 1."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $variant = if ($Residual -eq 0) { "control" } else { "residual" }
    $OutputDir = "outputs_pcbind_v9_${variant}_seed$Seed"
}

$DataFiles = @("Train335.pkl", "Test60.pkl", "Test287.pkl", "TestB25.pkl")
Push-Location $ProjectRoot
try {
    & $VenvPython (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
        --require-plm `
        --require-partner `
        --require-pair `
        --require-partner-encoder `
        @DataFiles
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython (Join-Path $ProjectRoot "smoke_test_pcbind_v9.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

if ($SmokeOnly -ne 0) {
    Write-Host "PC-BIND v9 residual smoke-only run complete."
    exit 0
}

# This runner is intentionally locked to grouped Train335 fold 1 with external
# evaluation disabled. A separate continuation runner should be created only if
# the frozen v9 gate passes.
& (Join-Path $ProjectRoot "run_seed.ps1") `
    -Seed $Seed `
    -OutputDir $OutputDir `
    -GroupedCV 1 `
    -CVGroupKey "complex_code" `
    -TestSets "Test60.pkl,Test287.pkl,TestB25.pkl" `
    -MaxFolds 1 `
    -SkipTestEval 1 `
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
    -PartnerLogitMode "residual" `
    -PartnerDeltaScale $AlphaValue `
    -PartnerResidueEncoder 1 `
    -PartnerEncoderLayers 1 `
    -PartnerTargetFusion "0.25" `
    -PartnerContrast 0 `
    -PairContactLoss 1 `
    -PairContactLossWeight $PairContactLossWeight `
    -PairContactPosWeight "8.0" `
    -PairContactHead "mlp" `
    -PairContactWarmupEpochs 5 `
    -PairMarginalConsistency 0 `
    -PairMarginalContrast 0 `
    -PairContactContrast 0 `
    -BatchSize $BatchSize
