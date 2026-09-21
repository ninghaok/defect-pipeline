$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available. Install Miniconda/Anaconda and reopen PowerShell."
}

conda env create -f .\environment.yml
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment may already exist; updating it instead."
    conda env update -n pipeline -f .\environment.yml --prune
}

conda run -n pipeline python -m pip install --upgrade pip
conda run -n pipeline python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 5080; use cu126 on older drivers
conda run -n pipeline python -m pip install -e ".[train]"
conda run -n pipeline python .\cli\check_local_installation.py

Write-Host "pipeline environment is ready."
