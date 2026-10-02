# Private shared recipe: the public entry points fix the treatment flag.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet(0, 1)][int]$PartnerTransport,
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
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "Missing Python environment: $VenvPython"
}

function Get-StrictExistingPath([string]$Path, [switch]$Directory) {
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ([bool]$item.PSIsContainer -ne [bool]$Directory) { throw "Unexpected path type: $Path" }
    # Resolve-Path alone does not dereference Windows junctions. Reject links
    # in this fixed cohort path, including ancestors, before reporting it.
    $part = $item
    while ($null -ne $part) {
        if (($part.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Filtered cohort paths must not traverse a link or junction: $($part.FullName)"
        }
        $part = if ($part -is [System.IO.DirectoryInfo]) { $part.Parent } else { $part.Directory }
    }
    return [System.IO.Path]::GetFullPath($item.FullName)
}

function Assert-Within([string]$Path, [string]$Root) {
    $prefix = $Root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $Path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the dedicated filtered cohort: $Path"
    }
}

function Get-VerifiedHash([string]$Path, [string]$Expected) {
    if ($Expected -notmatch '^[0-9a-fA-F]{64}$') { throw "Missing or invalid manifest SHA-256 for $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) { throw "SHA-256 mismatch: $Path" }
    return $actual
}

$dataDirectory = Get-StrictExistingPath (Join-Path $ProjectRoot "data\geo_filtered_v1") -Directory
$manifestPath = Get-StrictExistingPath (Join-Path $dataDirectory "cohort_manifest.json")
$foldPath = Get-StrictExistingPath (Join-Path $dataDirectory "grouped_folds_seed2101.json")
if (Test-Path -LiteralPath (Join-Path $dataDirectory "INCOMPLETE.txt")) {
    throw "Filtered cohort build is incomplete."
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.cohort_version -ne "geo_filtered_v1" -or
    $manifest.analysis_label -ne "filtered-cohort analysis" -or $manifest.status -ne "content_verified") {
    throw "Unexpected filtered-cohort manifest identity or status."
}
$declaredFiles = @("Train335.pkl", "Test287.pkl")
$manifestFiles = @($manifest.datasets.PSObject.Properties.Name)
if ($manifestFiles.Count -ne 2 -or @($manifestFiles | Where-Object { $_ -notin $declaredFiles }).Count -ne 0) {
    throw "This protocol declares exactly Train335.pkl and Test287.pkl."
}
$expectedCounts = @{ "Train335.pkl" = 334; "Test287.pkl" = 285 }
$datasetPaths = @{}
$datasetHashes = @{}
foreach ($filename in $declaredFiles) {
    $path = Get-StrictExistingPath (Join-Path $dataDirectory $filename)
    Assert-Within $path $dataDirectory
    $entry = $manifest.datasets.PSObject.Properties[$filename].Value
    if ($entry.sample_count -ne $expectedCounts[$filename]) { throw "Unexpected sample count for $filename" }
    if ($null -eq $entry.size_bytes -or (Get-Item -LiteralPath $path).Length -ne [int64]$entry.size_bytes) {
        throw "Manifest size_bytes mismatch: $path"
    }
    $recordedPath = Get-StrictExistingPath ([string]$entry.path)
    Assert-Within $recordedPath $dataDirectory
    if ($recordedPath -ne $path) { throw "Manifest path differs from the fixed $filename input." }
    $datasetPaths[$filename] = $path
    $datasetHashes[$filename] = Get-VerifiedHash $path ([string]$entry.sha256)
}
$foldEntry = $manifest.split_manifests.PSObject.Properties["2101"].Value
if ($null -eq $foldEntry) { throw "No grouped fold manifest is declared for seed 2101." }
$recordedFoldPath = Get-StrictExistingPath ([string]$foldEntry.path)
Assert-Within $recordedFoldPath $dataDirectory
if ($recordedFoldPath -ne $foldPath) { throw "Manifest declares a different grouped-fold path." }
if ($null -eq $foldEntry.size_bytes -or (Get-Item -LiteralPath $foldPath).Length -ne [int64]$foldEntry.size_bytes) {
    throw "Manifest size_bytes mismatch: $foldPath"
}
$foldHash = Get-VerifiedHash $foldPath ([string]$foldEntry.sha256)
$foldManifest = Get-Content -LiteralPath $foldPath -Raw | ConvertFrom-Json
if ($foldManifest.seed -ne 2101 -or $foldManifest.num_folds -ne 5 -or
    $foldManifest.group_key -ne "complex_code" -or $foldManifest.sample_count -ne 334 -or
    $foldManifest.train_sha256 -ne $datasetHashes["Train335.pkl"]) {
    throw "Grouped-fold manifest does not identify this filtered training cohort and seed."
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $arm = if ($PartnerTransport -eq 1) { "ot" } else { "noot" }
    $OutputDir = Join-Path $ProjectRoot "outputs_pcbind_${arm}_filtered_seed2101_fold$MaxFolds"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Get-Location).Path $OutputDir
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if ($OutputDir -eq $dataDirectory -or $OutputDir.StartsWith($dataDirectory + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Training output must be outside the filtered cohort."
}
if (-not $DryRun -and -not $ValidateOnly -and (Test-Path -LiteralPath $OutputDir)) {
    if (-not (Test-Path -LiteralPath $OutputDir -PathType Container) -or
        @(Get-ChildItem -LiteralPath $OutputDir -Force).Count -ne 0) {
        throw "Refusing nonempty training output directory: $OutputDir"
    }
}

# These seven facts precede every Python launch, including prerequisite checks.
Write-Host "Resolved data directory: $dataDirectory"
Write-Host "Cohort manifest: $manifestPath"
Write-Host "Declared files: Train335.pkl, Test287.pkl"
Write-Host "Training pickle: $($datasetPaths['Train335.pkl'])"
Write-Host "Grouped fold manifest: $foldPath"
Write-Host ("External evaluation: " + $(if ($SkipTestEval -eq 1) { "skipped" } else { "Test287.pkl only" }))
Write-Host ("Partner transport: " + $(if ($PartnerTransport -eq 1) { "enabled" } else { "disabled" }))

$prerequisiteArguments = @(
    (Join-Path $ProjectRoot "check_pcbind_prereqs.py"),
    "--data-dir", $dataDirectory, "--cohort-manifest", $manifestPath,
    "--fold-manifest", $foldPath, "--seed", "2101",
    "--require-plm", "--require-partner", "--require-pair", "--require-partner-encoder",
    "--min-partner-encoder-coverage", "1.0", "Train335.pkl", "Test287.pkl"
)
if (-not $ValidateOnly) {
    $prerequisiteArguments += @("--provenance-output", (Join-Path $OutputDir "prerequisite_provenance.json"))
}
$recipe = @{
    Seed = 2101; OutputDir = $OutputDir; DataDir = $dataDirectory
    CohortManifest = $manifestPath; FoldManifest = $foldPath
    GroupedCV = 1; CVGroupKey = "complex_code"; TestSets = "Test287.pkl"
    TwoHead = 1; RankLossWeight = "0.55"; RankFusion = "0.35"; RankWarmupEpochs = 3
    ModelDropout = "0.25"; EdgeDropout = "0.06"; WeightDecay = "3e-4"
    PartnerConditioning = 1; PartnerTopK = 16; PartnerDirectFusion = 0
    PartnerLogitMode = "standard"; PartnerResidueEncoder = 1; PartnerEncoderLayers = 1
    PartnerTargetFusion = "0.25"; PartnerContrast = 0; PairContactLoss = 0
    PairMarginalConsistency = 0; PairMarginalContrast = 0; PairContactContrast = 0
    PartnerTransport = $PartnerTransport; TransportSinkhornIters = 5; TransportTau = "0.1"
    TransportDustbin = 1; TransportTopK = 32; TransportEntropyWeight = "0.01"
    TransportSparsityWeight = "0.005"; TransportLogitFusion = "0.3"
    BatchSize = $BatchSize; MaxFolds = $MaxFolds; SkipTestEval = $SkipTestEval
    UsePlm = 1; UseAuxPlm = 1; PartnerContactAux = 1; PatchLabels = 0
    SelectionMetric = "mcc"; SelectionAuprWeight = "0.35"
    DryRun = $DryRun; ValidateOnly = $ValidateOnly
}
if ($DryRun) {
    $settings = & (Join-Path $ProjectRoot "run_seed.ps1") @recipe | ConvertFrom-Json
    $settings | Add-Member -NotePropertyName CohortManifestSHA256 -NotePropertyValue $manifestHash
    $settings | Add-Member -NotePropertyName FoldManifestSHA256 -NotePropertyValue $foldHash
    $settings | Add-Member -NotePropertyName TrainingPickleSHA256 -NotePropertyValue $datasetHashes["Train335.pkl"]
    $settings | Add-Member -NotePropertyName ExternalPickleSHA256 -NotePropertyValue $datasetHashes["Test287.pkl"]
    $settings | Add-Member -NotePropertyName PrerequisiteArguments -NotePropertyValue (ConvertTo-Json -InputObject @($prerequisiteArguments) -Compress)
    $settings | ConvertTo-Json
    return
}

# Exactly one prerequisite invocation; both files are checked even when external
# evaluation is skipped. ValidateOnly does not create provenance/output files.
if (-not $ValidateOnly) {
    [System.IO.Directory]::CreateDirectory($OutputDir) | Out-Null
    $launcherDatasets = [ordered]@{}
    foreach ($filename in $declaredFiles) {
        $entry = $manifest.datasets.PSObject.Properties[$filename].Value
        $launcherDatasets[$filename] = [ordered]@{
            path = $datasetPaths[$filename]; sha256 = $datasetHashes[$filename]
            size_bytes = (Get-Item -LiteralPath $datasetPaths[$filename]).Length
            sample_count = $entry.sample_count
        }
    }
    $launcherProvenance = [ordered]@{
        schema_version = 1; component = "filtered_v1_launcher"
        created_utc = [DateTime]::UtcNow.ToString("o"); analysis_label = "filtered-cohort analysis"
        resolved_data_dir = $dataDirectory; declared_files = $declaredFiles
        cohort_manifest = @{ path = $manifestPath; sha256 = $manifestHash; size_bytes = (Get-Item -LiteralPath $manifestPath).Length }
        datasets = $launcherDatasets
        fold_manifest = @{ path = $foldPath; sha256 = $foldHash; size_bytes = (Get-Item -LiteralPath $foldPath).Length }
        seed = $Seed; max_folds = $MaxFolds; grouped_cv = $true; cv_group_key = "complex_code"
        partner_transport = ($PartnerTransport -eq 1); skip_test_eval = ($SkipTestEval -eq 1)
        external_test_sets = @("Test287.pkl"); output_dir = $OutputDir
    }
    $provenanceFile = [System.IO.File]::Open((Join-Path $OutputDir "launcher_provenance.json"),
        [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
    $writer = [System.IO.StreamWriter]::new($provenanceFile, [System.Text.UTF8Encoding]::new($false))
    try { $writer.WriteLine(($launcherProvenance | ConvertTo-Json -Depth 12)) } finally { $writer.Dispose() }
}
Push-Location $ProjectRoot
try {
    & $VenvPython @prerequisiteArguments
    if ($LASTEXITCODE -ne 0) { throw "Filtered-cohort prerequisite check failed: $LASTEXITCODE" }
    & (Join-Path $ProjectRoot "run_seed.ps1") @recipe
} finally {
    Pop-Location
}
