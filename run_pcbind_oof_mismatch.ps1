param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$OutputDir = "",
    [string]$BlendAlpha = "0.10",
    [int]$BootstrapReplicates = 500,
    [int]$BatchSize = 4,
    [int]$ContactHardK = 10,
    [string]$ControlSummary = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $leaf = Split-Path -Leaf $RunDir
    $OutputDir = "${leaf}_oof_mismatch"
}

Push-Location $ProjectRoot
try {
    $Arguments = @(
        (Join-Path $ProjectRoot "evaluate_pcbind_oof_mismatch.py"),
        "--run-dir", $RunDir,
        "--output-dir", $OutputDir,
        "--blend-alpha", $BlendAlpha,
        "--bootstrap-replicates", $BootstrapReplicates,
        "--batch-size", $BatchSize,
        "--contact-hard-k", $ContactHardK
    )
    if (-not [string]::IsNullOrWhiteSpace($ControlSummary)) {
        $Arguments += @("--control-summary", $ControlSummary)
    }
    & $VenvPython @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
