param(
    [int]$Seed = 2101,
    [string]$OutputDir = "",
    [int]$BatchSize = 1,
    [int]$SmokeOnly = 0,
    [ValidateRange(1, 5)][int]$MaxFolds = 5,
    [int]$SkipTestEval = 0,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_pcbind_no_ot_seed$Seed"
}
# Same input features/training recipe as the OT pilot, transport disabled.
# This is a branch-removal control; use the ablation for a capacity-only test.
& (Join-Path $ProjectRoot "run_pcbind_ot_pilot.ps1") `
    -Seed $Seed -OutputDir $OutputDir -BatchSize $BatchSize `
    -SmokeOnly $SmokeOnly -MaxFolds $MaxFolds -SkipTestEval $SkipTestEval `
    -PartnerTransport 0 -DryRun:$DryRun
