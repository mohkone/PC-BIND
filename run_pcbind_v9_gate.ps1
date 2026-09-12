param(
    [string]$ControlRunDir = "outputs_pcbind_v9_control_seed2111",
    [string]$TreatmentRunDir = "outputs_pcbind_v9_residual_seed2111",
    [int]$BootstrapReplicates = 500,
    [int]$BatchSize = 4,
    [string]$Output = "pcbind_v9_gate_result.json"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
$ControlAuditDir = "${ControlRunDir}_pathway_audit"
$TreatmentAuditDir = "${TreatmentRunDir}_pathway_audit"

Push-Location $ProjectRoot
try {
    & (Join-Path $ProjectRoot "run_pcbind_pathway_audit.ps1") `
        -RunDir $ControlRunDir `
        -OutputDir $ControlAuditDir `
        -BootstrapReplicates $BootstrapReplicates `
        -BatchSize $BatchSize
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & (Join-Path $ProjectRoot "run_pcbind_pathway_audit.ps1") `
        -RunDir $TreatmentRunDir `
        -OutputDir $TreatmentAuditDir `
        -BootstrapReplicates $BootstrapReplicates `
        -BatchSize $BatchSize
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $VenvPython (Join-Path $ProjectRoot "evaluate_pcbind_v9_gate.py") `
        --control-summary (Join-Path $ControlAuditDir "pcbind_pathway_utilization_summary.json") `
        --treatment-summary (Join-Path $TreatmentAuditDir "pcbind_pathway_utilization_summary.json") `
        --output $Output
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
