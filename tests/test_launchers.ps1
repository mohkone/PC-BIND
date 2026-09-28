$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Read-Config([string]$Script, [hashtable]$Options = @{}) {
    $json = & (Join-Path $ProjectRoot $Script) @Options -DryRun 6>$null
    return ($json -join "`n" | ConvertFrom-Json)
}
function Assert-Equal($Actual, $Expected, [string]$Message) {
    if ($Actual -ne $Expected) { throw "$Message (actual=$Actual, expected=$Expected)" }
}

# Intentionally poison the caller; each launch must be isolated and reversible.
$oldTransport = $env:PPI_PARTNER_TRANSPORT
$oldDisablePlm = $env:PPI_DISABLE_PLM
$oldLocation = Get-Location
try {
    $env:PPI_PARTNER_TRANSPORT = "1"
    $env:PPI_DISABLE_PLM = "1"
    Set-Location $env:TEMP
    $callerLocation = (Get-Location).Path
    $base = Read-Config "run_seed.ps1" @{
        PartnerDeltaScale = "0.2"; PairContactHead = "mlp"
        PairMarginalContrastWeight = "0.07"; SkipTestEval = 1
    }
    Assert-Equal $base.PPI_PARTNER_TRANSPORT "0" "Default run inherited transport"
    Assert-Equal $base.PPI_DISABLE_PLM $null "Default run inherited PLM override"
    Assert-Equal $base.PPI_PARTNER_DELTA_SCALE "0.2" "Missing residual option"
    Assert-Equal $base.PPI_PAIR_CONTACT_HEAD "mlp" "Missing pair-head option"
    Assert-Equal $base.PPI_PAIR_MARGINAL_CONTRAST_WEIGHT "0.07" "Missing contrast option"
    Assert-Equal $base.PPI_SKIP_TEST_EVAL "1" "Development run would evaluate tests"
    Assert-Equal $env:PPI_PARTNER_TRANSPORT "1" "Caller transport was changed"
    Assert-Equal $env:PPI_DISABLE_PLM "1" "Caller PLM setting was changed"
    Assert-Equal (Get-Location).Path $callerLocation "Caller working directory was changed"

    $ot = Read-Config "run_pcbind_ot_pilot.ps1"
    $control = Read-Config "run_pcbind_no_ot_control.ps1"
    $ablation = Read-Config "run_pcbind_ot_ablation.ps1"
    Assert-Equal $ot.PPI_PARTNER_TRANSPORT "1" "OT runner disabled transport"
    Assert-Equal $control.PPI_PARTNER_TRANSPORT "0" "Control enabled transport"
    Assert-Equal $ablation.PPI_TRANSPORT_SINKHORN_ITERS "0" "Capacity ablation still projects"
    foreach ($property in $ot.PSObject.Properties) {
        $name = $property.Name
        if ($name -notin @("PPI_OUTPUT_DIR", "PPI_PARTNER_TRANSPORT")) {
            Assert-Equal $control.$name $property.Value "Control differs in $name"
        }
        if ($name -notin @("PPI_OUTPUT_DIR", "PPI_TRANSPORT_SINKHORN_ITERS")) {
            Assert-Equal $ablation.$name $property.Value "Ablation differs in $name"
        }
    }
    if ($ot.PPI_OUTPUT_DIR -eq $control.PPI_OUTPUT_DIR) { throw "Control overwrites OT output" }
    $oneFold = Read-Config "run_seed_onefold.ps1"
    Assert-Equal $oneFold.PPI_MAX_FOLDS "1" "One-fold entry point runs five folds"
    Write-Host "Launcher integrity checks passed."
} finally {
    $env:PPI_PARTNER_TRANSPORT = $oldTransport
    $env:PPI_DISABLE_PLM = $oldDisablePlm
    Set-Location $oldLocation
}
