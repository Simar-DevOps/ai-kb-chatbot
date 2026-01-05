# scripts/run_all.ps1
# Usage (from repo root):
#   .\scripts\run_all.ps1
#
# Purpose:
# - Standardize the Week 3 regression workflow into one command.
# - Step 1 (eval generation) is currently missing because evals/run_all.py was deleted/untracked.
# - Steps 2–4 run with your existing scripts + outputs.

Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "`n=== RUN ALL (Week 3 Regression) ===" -ForegroundColor Cyan

# Ensure output dirs exist
New-Item -ItemType Directory -Force -Path ".\evals\results" | Out-Null
New-Item -ItemType Directory -Force -Path ".\evals\analysis" | Out-Null

Write-Host "`n[1/4] Generate a NEW run_*.csv in evals/results" -ForegroundColor Cyan
Write-Host "MISSING STEP: Previously you ran: python .\evals\run_all.py" -ForegroundColor Yellow
Write-Host "That file no longer exists and was not tracked in git." -ForegroundColor Yellow
Write-Host "Action: restore or recreate the eval generator script, then replace this block with that command." -ForegroundColor Yellow
exit 1

Write-Host "`n[2/4] Prefill/score latest run (Day 20)" -ForegroundColor Cyan
python .\evals\analysis\day20_prefill_scores.py

Write-Host "`n[3/4] Failure analysis + buckets (Day 19)" -ForegroundColor Cyan
python .\evals\analysis\day19_failure_analysis.py

Write-Host "`n[4/4] Show newest key outputs" -ForegroundColor Cyan
Get-ChildItem .\evals\results -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name, LastWriteTime, Length |
  Format-Table -AutoSize

Write-Host "`nDone. Review day18 quick checks + top_failures.md." -ForegroundColor Green
