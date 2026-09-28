$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$fixtureRoot = Join-Path $temporaryRoot ("pcbind_filtered_launchers_" + [Guid]::NewGuid().ToString('N'))
$oldDataDir = $env:PPI_DATA_DIR
$oldTransport = $env:PPI_PARTNER_TRANSPORT
$oldLocation = (Get-Location).Path

function Assert-Equal($Actual, $Expected, [string]$Message) {
    if ($Actual -ne $Expected) { throw "$Message (actual=$Actual, expected=$Expected)" }
}
function Assert-Fails([scriptblock]$Action, [string]$Pattern) {
    $failed = $false
    try { & $Action 6>$null | Out-Null } catch {
        $failed = $true
        if ($_.Exception.Message -notmatch $Pattern) { throw "Wrong failure: $($_.Exception.Message)" }
    }
    if (-not $failed) { throw "Expected failure matching $Pattern" }
}
function Hash-File([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Write-FixtureManifest {
    $fold = @{ schema_version = 1; seed = 2101; num_folds = 5; group_key = "complex_code";
        sample_count = 334; train_sha256 = (Hash-File $trainPath) }
    $fold | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $foldPath -Encoding utf8
    $manifest = @{
        schema_version = 1; cohort_version = "geo_filtered_v1"; analysis_label = "filtered-cohort analysis";
        status = "content_verified"; datasets = @{
            "Train335.pkl" = @{ path = $trainPath; sha256 = (Hash-File $trainPath); sample_count = 334; size_bytes = (Get-Item -LiteralPath $trainPath).Length }
            "Test287.pkl" = @{ path = $testPath; sha256 = (Hash-File $testPath); sample_count = 285; size_bytes = (Get-Item -LiteralPath $testPath).Length }
        }
        split_manifests = @{ "2101" = @{ path = $foldPath; sha256 = (Hash-File $foldPath); size_bytes = (Get-Item -LiteralPath $foldPath).Length } }
    }
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
}
function Read-FilteredConfig([string]$Script, [hashtable]$Options = @{}) {
    return (& (Join-Path $fixtureRoot $Script) @Options -DryRun 6>$null | ConvertFrom-Json)
}

try {
    [System.IO.Directory]::CreateDirectory((Join-Path $fixtureRoot '.venv\Scripts')) | Out-Null
    $dataPath = Join-Path $fixtureRoot 'data\geo_filtered_v1'
    [System.IO.Directory]::CreateDirectory($dataPath) | Out-Null
    foreach ($script in @('run_seed.ps1', 'invoke_filtered_v1.ps1',
                          'run_pcbind_ot_filtered_v1_pilot.ps1', 'run_pcbind_noot_filtered_v1_control.ps1')) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $script) -Destination (Join-Path $fixtureRoot $script)
    }
    # Dry-run tests never execute this placeholder or load these tiny fake pickles.
    Set-Content -LiteralPath (Join-Path $fixtureRoot '.venv\Scripts\python.exe') -Value 'must not execute'
    $trainPath = Join-Path $dataPath 'Train335.pkl'
    $testPath = Join-Path $dataPath 'Test287.pkl'
    $manifestPath = Join-Path $dataPath 'cohort_manifest.json'
    $foldPath = Join-Path $dataPath 'grouped_folds_seed2101.json'
    Set-Content -LiteralPath $trainPath -Value 'training fixture'
    Set-Content -LiteralPath $testPath -Value 'external fixture'
    Write-FixtureManifest
    $env:PPI_DATA_DIR = 'inherited-wrong-cohort'
    $env:PPI_PARTNER_TRANSPORT = 'unexpected'
    Set-Location $temporaryRoot
    $ot = Read-FilteredConfig 'run_pcbind_ot_filtered_v1_pilot.ps1'
    $control = Read-FilteredConfig 'run_pcbind_noot_filtered_v1_control.ps1'
    Assert-Equal $ot.PPI_DATA_DIR $dataPath 'Dedicated data path differs'
    Assert-Equal $ot.PPI_TEST_SETS 'Test287.pkl' 'Legacy external sets leaked in'
    Assert-Equal $ot.PPI_SKIP_TEST_EVAL '1' 'Default external evaluation is active'
    Assert-Equal $ot.PPI_MAX_FOLDS '1' 'Default pilot runs too many folds'
    Assert-Equal $ot.PPI_PARTNER_TRANSPORT '1' 'OT treatment disabled'
    Assert-Equal $control.PPI_PARTNER_TRANSPORT '0' 'Control enabled transport'
    Assert-Equal $env:PPI_DATA_DIR 'inherited-wrong-cohort' 'Caller data setting changed'
    Assert-Equal $env:PPI_PARTNER_TRANSPORT 'unexpected' 'Caller treatment setting changed'
    Assert-Equal (Get-Location).Path $temporaryRoot.TrimEnd('\') 'Caller directory changed'
    foreach ($property in $ot.PSObject.Properties) {
        if ($property.Name -notin @('PPI_OUTPUT_DIR', 'PPI_PARTNER_TRANSPORT', 'PrerequisiteArguments')) {
            Assert-Equal $control.($property.Name) $property.Value "Unmatched recipe: $($property.Name)"
        }
    }
    if ($ot.PPI_OUTPUT_DIR -eq $control.PPI_OUTPUT_DIR) { throw 'Both arms use the same output.' }
    if (Test-Path -LiteralPath $ot.PPI_OUTPUT_DIR) { throw 'Dry-run created an output directory.' }
    $pythonArgs = @($ot.PythonArguments | ConvertFrom-Json)
    foreach ($pair in @(@('--data-dir', $dataPath), @('--cohort-manifest', $manifestPath), @('--fold-manifest', $foldPath))) {
        $index = [Array]::IndexOf($pythonArgs, $pair[0])
        if ($index -lt 0) { throw "Missing explicit Python option: $($pair[0])" }
        Assert-Equal $pythonArgs[$index + 1] $pair[1] 'Explicit Python path mismatch'
    }
    $prereq = @($ot.PrerequisiteArguments | ConvertFrom-Json)
    $coverageIndex = [Array]::IndexOf($prereq, '--min-partner-encoder-coverage')
    Assert-Equal $prereq[$coverageIndex + 1] '1.0' 'Prerequisite coverage is relaxed'
    Assert-Equal @($prereq | Where-Object { $_ -eq 'Test287.pkl' }).Count 1 'External prerequisite repeated/missing'
    if (@($prereq | Where-Object { $_ -in @('Test60.pkl', 'TestB25.pkl', 'Test70.pkl', 'TestUB25.pkl') }).Count) {
        throw 'Legacy data appeared in the dedicated prerequisite call.'
    }
    $validateConfig = Read-FilteredConfig 'run_pcbind_ot_filtered_v1_pilot.ps1' @{ ValidateOnly = $true }
    if ('--validate-only' -notin @($validateConfig.PythonArguments | ConvertFrom-Json)) { throw 'Validation-only flag missing.' }
    Assert-Fails { & (Join-Path $fixtureRoot 'run_pcbind_ot_filtered_v1_pilot.ps1') -Seed 999 -DryRun } '2101'
    Assert-Fails { & (Join-Path $fixtureRoot 'run_pcbind_ot_filtered_v1_pilot.ps1') -DataDir 'data/geo' -DryRun } 'DataDir'
    Assert-Fails { & (Join-Path $fixtureRoot 'run_seed.ps1') -DataDir $dataPath -CohortManifest $manifestPath -DryRun } 'together'
    # Same-size tampering specifically exercises the content-hash gate.
    Set-Content -LiteralPath $testPath -Value 'changed! fixture'
    Assert-Fails { Read-FilteredConfig 'run_pcbind_ot_filtered_v1_pilot.ps1' } 'SHA-256 mismatch|size_bytes mismatch'
    Write-FixtureManifest
    Add-Content -LiteralPath $foldPath -Value ' '
    Assert-Fails { Read-FilteredConfig 'run_pcbind_ot_filtered_v1_pilot.ps1' } 'size_bytes mismatch'
    Write-FixtureManifest
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $manifest.datasets.'Test287.pkl'.path = $trainPath
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Assert-Fails { Read-FilteredConfig 'run_pcbind_ot_filtered_v1_pilot.ps1' } 'Manifest path differs'
    Write-FixtureManifest
    $nonempty = Join-Path $fixtureRoot 'already-used'
    [System.IO.Directory]::CreateDirectory($nonempty) | Out-Null
    Set-Content -LiteralPath (Join-Path $nonempty 'keep.txt') -Value 'existing result'
    Assert-Fails { & (Join-Path $fixtureRoot 'run_pcbind_ot_filtered_v1_pilot.ps1') -OutputDir $nonempty } 'nonempty'
    Assert-Equal (Get-Content -LiteralPath (Join-Path $nonempty 'keep.txt')) 'existing result' 'Output overwritten'
    Write-Host 'Filtered launcher checks passed (synthetic fixtures; no Python or training launched).'
} finally {
    $env:PPI_DATA_DIR = $oldDataDir
    $env:PPI_PARTNER_TRANSPORT = $oldTransport
    Set-Location $oldLocation
    # Resolve and check the final target before deleting only this test fixture.
    $resolvedFixture = [System.IO.Path]::GetFullPath($fixtureRoot)
    $allowedPrefix = $temporaryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedFixture.StartsWith($allowedPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not ([System.IO.Path]::GetFileName($resolvedFixture)).StartsWith('pcbind_filtered_launchers_')) {
        throw "Unsafe fixture cleanup target: $resolvedFixture"
    }
    if (Test-Path -LiteralPath $resolvedFixture) { Remove-Item -LiteralPath $resolvedFixture -Recurse -Force }
}
