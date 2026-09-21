# Stand-alone YOLO-seg training with the lifecycle method. Example:
#   .\scripts\train_yolo_windows.ps1 -Category di_mian_detection
#   .\scripts\train_yolo_windows.ps1 -Category wa_yuan_detection -TrainNgLimit 60 -Epochs 100 -SkipTest
param(
    [Parameter(Mandatory=$true)][ValidateSet('qiumian_fupai','qiumian_xiepai','di_mian_detection','wa_yuan_detection')][string]$Category,
    [string]$DatasetRoot = "D:\dataset_523\dataset_523",
    [string]$Output = "",
    [int]$TrainOkLimit = 0,
    [int]$TrainNgLimit = 0,
    [int]$Epochs = 0,
    [int]$Batch = 0,
    [switch]$SkipTest,
    [string]$Python = "D:\conda_data\envs\pipeline\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if (-not $Output) { $Output = "C:\ninghao\results\train_yolo_${Category}_" + (Get-Date -Format "yyyyMMdd_HHmmss") }
$CliArgs = @(".\cli\train_yolo.py", "--category", $Category, "--dataset-root", $DatasetRoot, "--output", $Output)
if ($TrainOkLimit -gt 0) { $CliArgs += @("--train-ok-limit", $TrainOkLimit) }
if ($TrainNgLimit -gt 0) { $CliArgs += @("--train-ng-limit", $TrainNgLimit) }
if ($Epochs -gt 0) { $CliArgs += @("--epochs", $Epochs) }
if ($Batch -gt 0) { $CliArgs += @("--batch", $Batch) }
if ($SkipTest) { $CliArgs += "--skip-test" }
Write-Host "OUTPUT: $Output" -ForegroundColor Cyan
& $Python -u @CliArgs
if ($LASTEXITCODE -ne 0) { throw "train_yolo failed ($LASTEXITCODE)" }
