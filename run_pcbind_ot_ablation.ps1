param(
    [int]$Seed = 2101,
    [string]$OutputDir = "",
    [int]$BatchSize = 1,
    [double]$TransportTau = 0.1,
    [double]$TransportLogitFusion = 0.3,
    [ValidateRange(1, 5)][int]$MaxFolds = 5,
    [int]$SkipTestEval = 0,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_pcbind_ot_ablation_seed$Seed"
}
# Preserve temperature, features, regularizers, dustbin and fusion. Only disable
# the capacity projection. Use the same tau/fusion overrides in both conditions.
& (Join-Path $ProjectRoot "run_pcbind_ot_pilot.ps1") `
    -Seed $Seed -OutputDir $OutputDir -BatchSize $BatchSize `
    -MaxFolds $MaxFolds -SkipTestEval $SkipTestEval -PartnerTransport 1 `
    -TransportSinkhornIters 0 -TransportTau $TransportTau `
    -TransportLogitFusion $TransportLogitFusion -DryRun:$DryRun
