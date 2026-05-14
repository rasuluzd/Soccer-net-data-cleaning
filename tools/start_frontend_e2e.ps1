# PowerShell helper to bring up the full ForzaSearch end-to-end stack
# AFTER Docker Desktop is running (manual prerequisite).
#
# Usage:
#   1. Start Docker Desktop manually (Windows task bar -> right-click whale icon -> Start)
#   2. Wait until "Docker Desktop is running"
#   3. Run this script:  pwsh tools/start_frontend_e2e.ps1
#
# What it does:
#   1. Starts the Elasticsearch 8.17 container (single-node, no security)
#   2. Waits for ES to accept connections on :9200
#   3. Runs `npm run ingest --force` (rebuilds the index from
#      frontend/.../matches/*/kamp.json including the Chelsea-Liverpool
#      match exported by tools/export_to_frontend.py)
#   4. Starts `npm run dev` so frontend is reachable on http://localhost:3000

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$front = Join-Path $root "frontend\forzasearch-final"

Write-Host "[1/4] Starting Elasticsearch container ..." -ForegroundColor Cyan
$existing = docker ps -a --format "{{.Names}}" | Select-String -Pattern "^forzasearch-es$"
if ($existing) {
    Write-Host "  -> existing 'forzasearch-es' container found, restarting"
    docker start forzasearch-es | Out-Null
} else {
    docker run -d --name forzasearch-es `
        -p 9200:9200 `
        -e "discovery.type=single-node" `
        -e "xpack.security.enabled=false" `
        -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" `
        docker.elastic.co/elasticsearch/elasticsearch:8.17.0 | Out-Null
    Write-Host "  -> created and started 'forzasearch-es'"
}

Write-Host "[2/4] Waiting for ES to accept connections on :9200 ..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:9200" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { throw "Elasticsearch did not become ready within 120 seconds" }
Write-Host "  -> ES ready" -ForegroundColor Green

Write-Host "[3/4] Running ingest (this rebuilds the search index) ..." -ForegroundColor Cyan
Push-Location $front
npm run ingest -- --force
Pop-Location

Write-Host "[4/4] Starting Next.js dev server (Ctrl+C to stop) ..." -ForegroundColor Cyan
Push-Location $front
Write-Host "  -> Once 'Ready in <ms>' shows, open http://localhost:3000" -ForegroundColor Yellow
Write-Host "  -> Pick 'Chelsea 1-2 Liverpool' from the dropdown and search e.g. 'Sturridge goal' or 'free kick'"
npm run dev
Pop-Location
