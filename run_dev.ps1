# BioCoworker Launcher Script for Windows PowerShell
# This starts the python FastAPI backend and the React + Electron frontend concurrently.

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "        Launching BioCoworker App...       " -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# Check if port 8989 is occupied by a previous run or orphan process
$PortCheck = Get-NetTCPConnection -LocalPort 8989 -ErrorAction SilentlyContinue
if ($PortCheck) {
    Write-Host "Port 8989 is currently occupied by process ID: $($PortCheck.OwningProcess[0]). Terminating old process..." -ForegroundColor Cyan
    foreach ($procId in $PortCheck.OwningProcess) {
        taskkill /F /T /PID $procId 2>$null
    }
    Start-Sleep -Seconds 1
}

# 1. Start Python FastAPI backend in a new job or background process
Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8989..." -ForegroundColor Yellow
$BackendProcess = Start-Process uv -ArgumentList "run python run_backend.py" -PassThru -NoNewWindow

# 2. Wait for backend to start up
Start-Sleep -Seconds 3

# 3. Start Electron + React Frontend
Write-Host "[2/2] Launching React + Electron Frontend..." -ForegroundColor Yellow
Start-Process cmd.exe -ArgumentList "/c npm run electron:dev" -WorkingDirectory "frontend" -NoNewWindow -Wait

# After frontend closes, stop the backend process
Write-Host "Closing application backend..." -ForegroundColor Yellow
if ($BackendProcess) {
    # Terminate the full process tree (including uv, python, and uvicorn subprocesses)
    taskkill /F /T /PID $BackendProcess.Id 2>$null
}
Write-Host "BioCoworker exited successfully." -ForegroundColor Green
