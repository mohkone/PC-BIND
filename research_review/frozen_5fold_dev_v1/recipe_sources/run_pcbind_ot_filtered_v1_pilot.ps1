[CmdletBinding()]
param(
    [ValidateSet(2101)][int]$Seed = 2101,
    [string]$OutputDir = "",
    [ValidateRange(1, 5)][int]$MaxFolds = 1,
    [ValidateSet(0, 1)][int]$SkipTestEval = 1,
    [ValidateRange(1, 1024)][int]$BatchSize = 1,
    [switch]$DryRun,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ProjectRoot "invoke_filtered_v1.ps1") -PartnerTransport 1 @PSBoundParameters
