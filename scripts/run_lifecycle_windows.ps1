# Closed-loop simulation for the four categories: pretrained detector -> YOLO-seg milestones -> promotion.
# Rerunning with the same -RunName resumes (already ingested images are skipped); change ratios -> new RunName.
# NG ratio = NG share of the stream (0.10 = 10 % NG). -1 (default) = every available NG. See docs/WINDOWS_SETUP.md for the counts.
param(
    [ValidateRange(-1.0,0.999999)][double]$QiumianFupaiNgRatio = -1,
    [ValidateRange(-1.0,0.999999)][double]$QiumianXiepaiNgRatio = -1,
    [ValidateRange(-1.0,0.999999)][double]$DiMianNgRatio = -1,
    [ValidateRange(-1.0,0.999999)][double]$WaiYuanNgRatio = -1,
    [string]$RunName = "lifecycle_seg",
    [string]$DatasetRoot = "D:\dataset_523\dataset_523",
    [string[]]$Categories = @('qiumian_fupai','qiumian_xiepai','di_mian_detection','wa_yuan_detection'),
    [string]$Python = "D:\conda_data\envs\pipeline\python.exe"
)
$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PIPELINE_RESULTS_ROOT = Join-Path $Project ("results\" + $RunName)
$env:PIPELINE_PRETRAINED_CACHE_ROOT = Join-Path $env:PIPELINE_RESULTS_ROOT "pretrained_cache"
$env:PIPELINE_PRETRAINED_RESULTS_ROOT = Join-Path $env:PIPELINE_RESULTS_ROOT "pretrained_artifacts"
$Manifest = Join-Path $env:PIPELINE_RESULTS_ROOT "stream_manifest.json"
$Inv = [Globalization.CultureInfo]::InvariantCulture
function RatioArg([string]$Category, [double]$Ratio) { if ($Ratio -lt 0) { "$Category=all" } else { "$Category=" + $Ratio.ToString($Inv) } }
$RatioArguments = @(
    "--ng-ratio", (RatioArg "qiumian_fupai" $QiumianFupaiNgRatio),
    "--ng-ratio", (RatioArg "qiumian_xiepai" $QiumianXiepaiNgRatio),
    "--ng-ratio", (RatioArg "di_mian_detection" $DiMianNgRatio),
    "--ng-ratio", (RatioArg "wa_yuan_detection" $WaiYuanNgRatio)
)
& $Python -u .\cli\prepare_simulation_streams.py --scenario simplified_lifecycle --output-root $env:PIPELINE_RESULTS_ROOT `
    --dataset-root $DatasetRoot --reference-ok 32 --calibration-ok 200 --bank-ok 200 --reserve qiumian_xiepai=32,100,100 --stream-mode stratified @RatioArguments
if ($LASTEXITCODE -ne 0) { throw "Failed to prepare the lifecycle stream" }
foreach ($Category in $Categories) {
    $Inbox = Join-Path (Join-Path $env:PIPELINE_RESULTS_ROOT "streams") $Category
    Write-Host "`n===== LIFECYCLE START: $Category =====" -ForegroundColor Cyan
    & $Python -u .\cli\run_lifecycle.py --category $Category --inbox $Inbox --initialization-manifest $Manifest --batch-id $RunName
    if ($LASTEXITCODE -ne 0) { throw "Lifecycle failed: $Category (rerun the same command to resume)" }
}
& $Python .\cli\audit.py
& $Python .\cli\report_metrics.py
Write-Host "`nALL LIFECYCLE RUNS COMPLETED: $env:PIPELINE_RESULTS_ROOT" -ForegroundColor Green
