# Restart the DCM Streamlit dashboard -- safely, and verify it is actually NEW code.
#
# WHY THIS EXISTS
#   Streamlit runs as a THREE-process tree: streamlit.exe -> python -> python (the
#   grandchild owns the listening socket). Killing "the process on port 8501" kills
#   only the grandchild; the parent survives, keeps the port, and keeps serving the
#   modules it imported at ITS start time. A plain `Invoke-WebRequest` then returns
#   HTTP 200 from the STALE instance, which reads as a successful restart.
#
#   That failure mode cost a debugging cycle: the page raised
#   "module 'src.analytics.deal_signals' has no attribute 'entity_leaderboard'" for a
#   function that had been on disk, imported fine in a shell, and passed its tests.
#   The server was 28 minutes older than the source file.
#
#   So: kill by COMMAND LINE (the whole tree), then verify the new server started
#   AFTER the newest .py under src/. HTTP 200 is not evidence of freshness.

param(
    [int]$Port = 8501,
    [string]$ProjectRoot = "D:\Python Projects\Daily_Cash_Market"
)

$ErrorActionPreference = "Stop"

function Get-DcmProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='streamlit.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -like "*Daily_Cash_Market*" -and $_.CommandLine -like "*streamlit*" }
}

$existing = @(Get-DcmProcesses)
if ($existing.Count -gt 0) {
    Write-Host "Stopping $($existing.Count) existing process(es): $($existing.ProcessId -join ', ')"
    foreach ($p in $existing) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
    }
    Start-Sleep -Seconds 3
}

$still = @(Get-DcmProcesses)
if ($still.Count -gt 0) {
    throw "Could not stop every dashboard process; still running: $($still.ProcessId -join ', ')"
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is still bound after killing the dashboard tree -- something else owns it."
}

# Newest source file: the bar the new server has to clear.
$newest = Get-ChildItem -Path (Join-Path $ProjectRoot "src") -Filter *.py -Recurse |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "Newest source: $($newest.Name) @ $($newest.LastWriteTime)"

Start-Process -FilePath (Join-Path $ProjectRoot "venv\Scripts\streamlit.exe") `
              -ArgumentList 'run', 'src\dashboard\app.py', "--server.port", $Port `
              -WorkingDirectory $ProjectRoot -WindowStyle Hidden

for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { break }
}
if (-not $conn) { throw "Dashboard did not bind port $Port within 30s." }

$srv = Get-Process -Id ($conn | Select-Object -First 1).OwningProcess
Write-Host "Serving PID $($srv.Id) started $($srv.StartTime)"

if ($srv.StartTime -lt $newest.LastWriteTime) {
    throw ("STALE SERVER: pid $($srv.Id) started $($srv.StartTime) which is BEFORE the newest " +
           "source file ($($newest.LastWriteTime)). It is serving old code.")
}

try {
    $code = (Invoke-WebRequest -Uri "http://localhost:$Port" -UseBasicParsing -TimeoutSec 20).StatusCode
    Write-Host "HTTP $code -- fresh server confirmed (started after every source change)."
} catch {
    throw "Server bound the port but did not answer: $_"
}
