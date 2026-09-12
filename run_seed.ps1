param(
    [int]$Seed = 1234,
    [string]$OutputDir = "",
    [int]$TwoHead = 1,
    [string]$AuxPlmDim = "",
    [string]$RankLossWeight = "0.55",
    [string]$ConsistencyWeight = "0.04",
    [string]$RankFusion = "0.35",
    [string]$ViewLogitFusion = "0.0",
    [string]$ModelDropout = "0.20",
    [string]$EdgeDropout = "0.04",
    [string]$WeightDecay = "2e-4",
    [int]$PartnerConditioning = 0,
    [int]$PartnerTopK = 16,
    [int]$PartnerDirectFusion = 0,
    [string]$PartnerLogitMode = "standard",
    [string]$PartnerDeltaScale = "1.0",
    [int]$PartnerResidueEncoder = 0,
    [int]$PartnerEncoderLayers = 1,
    [string]$PartnerTargetFusion = "0.25",
    [int]$PartnerContrast = 0,
    [string]$PartnerContrastWeight = "0.03",
    [string]$PartnerContrastMargin = "0.05",
    [int]$PartnerContrastWarmupEpochs = 0,
    [int]$PairContactLoss = 0,
    [string]$PairContactLossWeight = "0.06",
    [string]$PairContactPosWeight = "8.0",
    [string]$PairContactHead = "attention",
    [int]$PairContactWarmupEpochs = 0,
    [int]$PairMarginalConsistency = 0,
    [string]$PairMarginalConsistencyWeight = "0.02",
    [int]$PairMarginalConsistencyWarmupEpochs = 5,
    [int]$PairMarginalContrast = 0,
    [string]$PairMarginalContrastWeight = "0.02",
    [string]$PairMarginalContrastMargin = "0.03",
    [int]$PairMarginalContrastWarmupEpochs = 5,
    [int]$PairContactContrast = 0,
    [string]$PairContactContrastWeight = "0.005",
    [string]$PairContactContrastMargin = "0.03",
    [int]$PairContactContrastHardK = 10,
    [int]$PairContactContrastWarmupEpochs = 5,
    [int]$BatchSize = 1,
    [int]$MaxFolds = 5,
    [int]$SkipTestEval = 0,
    [int]$GroupedCV = 0,
    [string]$CVGroupKey = "complex_code",
    [string]$TestSets = "",
    [int]$UsePlm = 1,
    [int]$UseAuxPlm = 1,
    [int]$PartnerContactAux = 1,
    [int]$NoPlm = 0,
    [int]$NoAuxObjective = 0,
    [int]$PatchLabels = 0,
    [int]$PatchLabelSteps = 2,
    [string]$PatchLabelAlpha = "0.45",
    [string]$PatchLabelMaxUnlabeled = "0.45",
    [int]$RankWarmupEpochs = 0,
    [string]$SelectionMetric = "mcc",
    [string]$SelectionAuprWeight = "0.35"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found. Run .\setup_env.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "outputs_seed$Seed"
}

if ($NoPlm -ne 0) {
    $UsePlm = 0
    $UseAuxPlm = 0
}
if ($NoAuxObjective -ne 0) {
    $UseAuxPlm = 0
    $PartnerContactAux = 0
}

$env:PPI_SEED = "$Seed"
$env:PPI_OUTPUT_DIR = $OutputDir
$env:PPI_TWO_HEAD_BINDING = "$TwoHead"
$env:PPI_TWO_HEAD_RANK_LOSS_WEIGHT = "$RankLossWeight"
$env:PPI_TWO_HEAD_CONSISTENCY_WEIGHT = "$ConsistencyWeight"
$env:PPI_TWO_HEAD_RANK_FUSION = "$RankFusion"
$env:PPI_VIEW_LOGIT_FUSION = "$ViewLogitFusion"
$env:PPI_MODEL_DROPOUT = "$ModelDropout"
$env:PPI_MODEL_EDGE_DROPOUT = "$EdgeDropout"
$env:PPI_WEIGHT_DECAY = "$WeightDecay"
$env:PPI_PARTNER_CONDITIONING = "$PartnerConditioning"
$env:PPI_PARTNER_TOP_K = "$PartnerTopK"
$env:PPI_PARTNER_DIRECT_FUSION = "$PartnerDirectFusion"
$env:PPI_PARTNER_LOGIT_MODE = "$PartnerLogitMode"
$env:PPI_PARTNER_DELTA_SCALE = "$PartnerDeltaScale"
$env:PPI_PARTNER_RESIDUE_ENCODER = "$PartnerResidueEncoder"
$env:PPI_PARTNER_ENCODER_LAYERS = "$PartnerEncoderLayers"
$env:PPI_PARTNER_TARGET_FUSION = "$PartnerTargetFusion"
$env:PPI_PARTNER_CONTRAST = "$PartnerContrast"
$env:PPI_PARTNER_CONTRAST_WEIGHT = "$PartnerContrastWeight"
$env:PPI_PARTNER_CONTRAST_MARGIN = "$PartnerContrastMargin"
$env:PPI_PARTNER_CONTRAST_WARMUP_EPOCHS = "$PartnerContrastWarmupEpochs"
$env:PPI_PAIR_CONTACT_LOSS = "$PairContactLoss"
$env:PPI_PAIR_CONTACT_LOSS_WEIGHT = "$PairContactLossWeight"
$env:PPI_PAIR_CONTACT_POS_WEIGHT = "$PairContactPosWeight"
$env:PPI_PAIR_CONTACT_HEAD = "$PairContactHead"
$env:PPI_PAIR_CONTACT_WARMUP_EPOCHS = "$PairContactWarmupEpochs"
$env:PPI_PAIR_MARGINAL_CONSISTENCY = "$PairMarginalConsistency"
$env:PPI_PAIR_MARGINAL_CONSISTENCY_WEIGHT = "$PairMarginalConsistencyWeight"
$env:PPI_PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS = "$PairMarginalConsistencyWarmupEpochs"
$env:PPI_PAIR_MARGINAL_CONTRAST = "$PairMarginalContrast"
$env:PPI_PAIR_MARGINAL_CONTRAST_WEIGHT = "$PairMarginalContrastWeight"
$env:PPI_PAIR_MARGINAL_CONTRAST_MARGIN = "$PairMarginalContrastMargin"
$env:PPI_PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS = "$PairMarginalContrastWarmupEpochs"
$env:PPI_PAIR_CONTACT_CONTRAST = "$PairContactContrast"
$env:PPI_PAIR_CONTACT_CONTRAST_WEIGHT = "$PairContactContrastWeight"
$env:PPI_PAIR_CONTACT_CONTRAST_MARGIN = "$PairContactContrastMargin"
$env:PPI_PAIR_CONTACT_CONTRAST_HARD_K = "$PairContactContrastHardK"
$env:PPI_PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS = "$PairContactContrastWarmupEpochs"
$env:PPI_BATCH_SIZE = "$BatchSize"
$env:PPI_MAX_FOLDS = "$MaxFolds"
$env:PPI_SKIP_TEST_EVAL = "$SkipTestEval"
$env:PPI_GROUPED_CV = "$GroupedCV"
$env:PPI_CV_GROUP_KEY = "$CVGroupKey"
if ([string]::IsNullOrWhiteSpace($TestSets)) {
    Remove-Item Env:PPI_TEST_SETS -ErrorAction SilentlyContinue
} else {
    $env:PPI_TEST_SETS = "$TestSets"
}
$env:PPI_USE_PLM_FEATURES = "$UsePlm"
$env:PPI_USE_AUX_PLM_FEATURES = "$UseAuxPlm"
$env:PPI_PARTNER_CONTACT_AUX = "$PartnerContactAux"
if ($UsePlm -eq 0) {
    $env:PPI_PLM_DIM = "0"
    $env:PPI_PLM_FEATURE_MODE = "disabled"
} else {
    Remove-Item Env:PPI_PLM_DIM -ErrorAction SilentlyContinue
    Remove-Item Env:PPI_PLM_FEATURE_MODE -ErrorAction SilentlyContinue
}
if ($UseAuxPlm -eq 0) {
    $env:PPI_AUX_PLM_DIM = "0"
    $env:PPI_AUX_PLM_FEATURE_MODE = "disabled"
} else {
    Remove-Item Env:PPI_AUX_PLM_DIM -ErrorAction SilentlyContinue
    Remove-Item Env:PPI_AUX_PLM_FEATURE_MODE -ErrorAction SilentlyContinue
}
$env:PPI_TWO_HEAD_RANK_WARMUP_EPOCHS = "$RankWarmupEpochs"
$env:PPI_CHECKPOINT_SELECTION_METRIC = "$SelectionMetric"
$env:PPI_CHECKPOINT_SELECTION_AUPR_WEIGHT = "$SelectionAuprWeight"
$env:PPI_PATCH_LABEL_DISTRIBUTION = "$PatchLabels"
$env:PPI_PATCH_LABEL_STEPS = "$PatchLabelSteps"
$env:PPI_PATCH_LABEL_ALPHA = "$PatchLabelAlpha"
$env:PPI_PATCH_LABEL_MAX_UNLABELED = "$PatchLabelMaxUnlabeled"
if (-not [string]::IsNullOrWhiteSpace($AuxPlmDim)) {
    $env:PPI_AUX_PLM_DIM = "$AuxPlmDim"
}

Write-Host "Running seed $Seed -> $OutputDir"
Write-Host "PPI_TWO_HEAD_BINDING=$env:PPI_TWO_HEAD_BINDING"
Write-Host "PPI_TWO_HEAD_RANK_LOSS_WEIGHT=$env:PPI_TWO_HEAD_RANK_LOSS_WEIGHT"
Write-Host "PPI_TWO_HEAD_CONSISTENCY_WEIGHT=$env:PPI_TWO_HEAD_CONSISTENCY_WEIGHT"
Write-Host "PPI_TWO_HEAD_RANK_FUSION=$env:PPI_TWO_HEAD_RANK_FUSION"
Write-Host "PPI_VIEW_LOGIT_FUSION=$env:PPI_VIEW_LOGIT_FUSION"
Write-Host "PPI_MODEL_DROPOUT=$env:PPI_MODEL_DROPOUT"
Write-Host "PPI_MODEL_EDGE_DROPOUT=$env:PPI_MODEL_EDGE_DROPOUT"
Write-Host "PPI_WEIGHT_DECAY=$env:PPI_WEIGHT_DECAY"
Write-Host "PPI_PARTNER_CONDITIONING=$env:PPI_PARTNER_CONDITIONING"
if ($PartnerConditioning -ne 0) {
    Write-Host "PPI_PARTNER_TOP_K=$env:PPI_PARTNER_TOP_K"
    Write-Host "PPI_PARTNER_DIRECT_FUSION=$env:PPI_PARTNER_DIRECT_FUSION"
    Write-Host "PPI_PARTNER_LOGIT_MODE=$env:PPI_PARTNER_LOGIT_MODE"
    Write-Host "PPI_PARTNER_RESIDUE_ENCODER=$env:PPI_PARTNER_RESIDUE_ENCODER"
    if ($PartnerResidueEncoder -ne 0) {
        Write-Host "PPI_PARTNER_ENCODER_LAYERS=$env:PPI_PARTNER_ENCODER_LAYERS"
        Write-Host "PPI_PARTNER_TARGET_FUSION=$env:PPI_PARTNER_TARGET_FUSION"
    }
    if ($PartnerLogitMode -eq "delta" -or $PartnerLogitMode -eq "residual") {
        Write-Host "PPI_PARTNER_DELTA_SCALE=$env:PPI_PARTNER_DELTA_SCALE"
    }
}
Write-Host "PPI_PARTNER_CONTRAST=$env:PPI_PARTNER_CONTRAST"
if ($PartnerContrast -ne 0) {
    Write-Host "PPI_PARTNER_CONTRAST_WEIGHT=$env:PPI_PARTNER_CONTRAST_WEIGHT"
    Write-Host "PPI_PARTNER_CONTRAST_MARGIN=$env:PPI_PARTNER_CONTRAST_MARGIN"
    Write-Host "PPI_PARTNER_CONTRAST_WARMUP_EPOCHS=$env:PPI_PARTNER_CONTRAST_WARMUP_EPOCHS"
}
Write-Host "PPI_PAIR_CONTACT_LOSS=$env:PPI_PAIR_CONTACT_LOSS"
if ($PairContactLoss -ne 0) {
    Write-Host "PPI_PAIR_CONTACT_LOSS_WEIGHT=$env:PPI_PAIR_CONTACT_LOSS_WEIGHT"
    Write-Host "PPI_PAIR_CONTACT_POS_WEIGHT=$env:PPI_PAIR_CONTACT_POS_WEIGHT"
    Write-Host "PPI_PAIR_CONTACT_HEAD=$env:PPI_PAIR_CONTACT_HEAD"
    Write-Host "PPI_PAIR_CONTACT_WARMUP_EPOCHS=$env:PPI_PAIR_CONTACT_WARMUP_EPOCHS"
}
Write-Host "PPI_PAIR_MARGINAL_CONSISTENCY=$env:PPI_PAIR_MARGINAL_CONSISTENCY"
if ($PairMarginalConsistency -ne 0) {
    Write-Host "PPI_PAIR_MARGINAL_CONSISTENCY_WEIGHT=$env:PPI_PAIR_MARGINAL_CONSISTENCY_WEIGHT"
    Write-Host "PPI_PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS=$env:PPI_PAIR_MARGINAL_CONSISTENCY_WARMUP_EPOCHS"
}
Write-Host "PPI_PAIR_MARGINAL_CONTRAST=$env:PPI_PAIR_MARGINAL_CONTRAST"
if ($PairMarginalContrast -ne 0) {
    Write-Host "PPI_PAIR_MARGINAL_CONTRAST_WEIGHT=$env:PPI_PAIR_MARGINAL_CONTRAST_WEIGHT"
    Write-Host "PPI_PAIR_MARGINAL_CONTRAST_MARGIN=$env:PPI_PAIR_MARGINAL_CONTRAST_MARGIN"
    Write-Host "PPI_PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS=$env:PPI_PAIR_MARGINAL_CONTRAST_WARMUP_EPOCHS"
}
Write-Host "PPI_PAIR_CONTACT_CONTRAST=$env:PPI_PAIR_CONTACT_CONTRAST"
if ($PairContactContrast -ne 0) {
    Write-Host "PPI_PAIR_CONTACT_CONTRAST_WEIGHT=$env:PPI_PAIR_CONTACT_CONTRAST_WEIGHT"
    Write-Host "PPI_PAIR_CONTACT_CONTRAST_MARGIN=$env:PPI_PAIR_CONTACT_CONTRAST_MARGIN"
    Write-Host "PPI_PAIR_CONTACT_CONTRAST_HARD_K=$env:PPI_PAIR_CONTACT_CONTRAST_HARD_K"
    Write-Host "PPI_PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS=$env:PPI_PAIR_CONTACT_CONTRAST_WARMUP_EPOCHS"
}
Write-Host "PPI_BATCH_SIZE=$env:PPI_BATCH_SIZE"
Write-Host "PPI_MAX_FOLDS=$env:PPI_MAX_FOLDS"
Write-Host "PPI_SKIP_TEST_EVAL=$env:PPI_SKIP_TEST_EVAL"
Write-Host "PPI_GROUPED_CV=$env:PPI_GROUPED_CV"
if ($GroupedCV -ne 0) {
    Write-Host "PPI_CV_GROUP_KEY=$env:PPI_CV_GROUP_KEY"
}
if ($env:PPI_TEST_SETS) {
    Write-Host "PPI_TEST_SETS=$env:PPI_TEST_SETS"
}
Write-Host "PPI_USE_PLM_FEATURES=$env:PPI_USE_PLM_FEATURES"
Write-Host "PPI_USE_AUX_PLM_FEATURES=$env:PPI_USE_AUX_PLM_FEATURES"
Write-Host "PPI_PARTNER_CONTACT_AUX=$env:PPI_PARTNER_CONTACT_AUX"
if ($UsePlm -eq 0) {
    Write-Host "PPI_PLM_DIM=$env:PPI_PLM_DIM"
    Write-Host "PPI_PLM_FEATURE_MODE=$env:PPI_PLM_FEATURE_MODE"
}
if ($UseAuxPlm -eq 0) {
    Write-Host "PPI_AUX_PLM_DIM=$env:PPI_AUX_PLM_DIM"
    Write-Host "PPI_AUX_PLM_FEATURE_MODE=$env:PPI_AUX_PLM_FEATURE_MODE"
}
Write-Host "PPI_TWO_HEAD_RANK_WARMUP_EPOCHS=$env:PPI_TWO_HEAD_RANK_WARMUP_EPOCHS"
Write-Host "PPI_CHECKPOINT_SELECTION_METRIC=$env:PPI_CHECKPOINT_SELECTION_METRIC"
Write-Host "PPI_CHECKPOINT_SELECTION_AUPR_WEIGHT=$env:PPI_CHECKPOINT_SELECTION_AUPR_WEIGHT"
Write-Host "PPI_PATCH_LABEL_DISTRIBUTION=$env:PPI_PATCH_LABEL_DISTRIBUTION"
if ($PatchLabels -ne 0) {
    Write-Host "PPI_PATCH_LABEL_STEPS=$env:PPI_PATCH_LABEL_STEPS"
    Write-Host "PPI_PATCH_LABEL_ALPHA=$env:PPI_PATCH_LABEL_ALPHA"
    Write-Host "PPI_PATCH_LABEL_MAX_UNLABELED=$env:PPI_PATCH_LABEL_MAX_UNLABELED"
}
if ($env:PPI_AUX_PLM_DIM) {
    Write-Host "PPI_AUX_PLM_DIM=$env:PPI_AUX_PLM_DIM"
}

& $VenvPython (Join-Path $ProjectRoot "CROSS5FOLD_multi_test.py")
