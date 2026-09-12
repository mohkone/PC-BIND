param(
    [string[]]$Files = @(
        "Train335.pkl",
        "Test60.pkl",
        "Test287.pkl",
        "TestB25.pkl",
        "TestUB25.pkl"
    )
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Missing Python environment: $VenvPython"
}

Push-Location $ProjectRoot
try {
    Write-Host "Adding chain-aware partner geometry and sequences..."
    & $VenvPython (Join-Path $ProjectRoot "augment_with_pdb_coordinates.py") @Files
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Extracting partner ESM-2 35M embeddings..."
    & $VenvPython `
        (Join-Path $ProjectRoot "extract_plm_embeddings.py") `
        --partner `
        --model "facebook/esm2_t12_35M_UR50D" `
        @Files
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Extracting partner ESM-2 8M embeddings..."
    & $VenvPython `
        (Join-Path $ProjectRoot "extract_plm_embeddings.py") `
        --partner `
        --model "facebook/esm2_t6_8M_UR50D" `
        --feature-key "residue_plm_embedding_8m" `
        --model-key "residue_plm_model_8m" `
        @Files
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Validating PC-BIND v5 inputs..."
    $FullPartnerFiles = @($Files | Where-Object { $_ -ne "TestUB25.pkl" })
    if ($FullPartnerFiles.Count -gt 0) {
        & $VenvPython `
            (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
            --require-plm `
            --require-partner `
            --require-pair `
            --require-partner-encoder `
            @FullPartnerFiles
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if ($Files -contains "TestUB25.pkl") {
        # The frozen unbound benchmark contains 12 valid partner inputs and 13
        # target-pathway fallbacks. Guard the documented minimum coverage.
        & $VenvPython `
            (Join-Path $ProjectRoot "check_pcbind_prereqs.py") `
            --require-plm `
            --require-partner `
            --require-pair `
            --require-partner-encoder `
            --min-partner-encoder-coverage 0.48 `
            "TestUB25.pkl"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
finally {
    Pop-Location
}

Write-Host "PC-BIND v5 data preparation complete."
