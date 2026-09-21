# NG-count ablation for the four categories with a live console progress bar (no file redirection).
# Whole-val calibration, ROI for the outer ring, epoch 100, recall-first thresholds under an FPR cap of 0.2.
# Interrupted? Rerun with -Stamp <same stamp> and finished NG counts are skipped.
param(
  [string]$Python = "D:\conda_data\envs\pipeline\python.exe",
  [string]$DatasetRoot = "D:\dataset_523\dataset_523",
  [int]$Epochs = 100,
  [string]$Stamp = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
if (-not $Stamp) { $Stamp = Get-Date -Format "yyyyMMdd_HHmmss" }
$jobs = @(
  @{ c = "qiusaidimian";  counts = @(20,40,60,80);  extra = @() },
  @{ c = "qiumianxiepai"; counts = @(20,40,60,80);  extra = @() },
  @{ c = "qiumianfupai";  counts = @(20,40,60,80);  extra = @() },
  @{ c = "qiusaiwaiyuan"; counts = @(20,40,80,150); extra = @("--skip-full") }
)
foreach ($j in $jobs) {
  $run = "C:\ninghao\results\seg_ngcount_fullval_$($j.c)_$Stamp"
  Write-Host "`n===== $($j.c) -> $run =====" -ForegroundColor Cyan
  & $Python -u .\cli\yolo_seg_ng_count_ablation.py --dataset-root $DatasetRoot --run-root $run --classes $j.c `
      --counts @($j.counts | ForEach-Object { "$_" }) --epochs $Epochs --patience 0 --max-fpr 0.2 @($j.extra)
  if ($LASTEXITCODE -ne 0) { throw "$($j.c) failed; resume with: .\scripts\run_seg_ng_ablation_live.ps1 -Stamp $Stamp" }
}
Write-Host "`nALL DONE. Results: C:\ninghao\results\seg_ngcount_fullval_<category>_$Stamp" -ForegroundColor Green
