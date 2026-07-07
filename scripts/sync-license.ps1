# sync-license.ps1
# This script reads ecosystem.json, clones each repository into a temporary folder,
# copies the master LICENSE file into it, and pushes the changes back.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Get-Item $scriptDir).Parent.FullName
$configFile = Join-Path $rootDir "ecosystem.json"
$masterLicense = Join-Path $rootDir "LICENSE"

if (-not (Test-Path $configFile)) {
    Write-Error "ecosystem.json not found in $rootDir"
    exit 1
}

if (-not (Test-Path $masterLicense)) {
    Write-Error "Master LICENSE file not found in $rootDir"
    exit 1
}

$config = Get-Content $configFile | ConvertFrom-Json
$tempDir = Join-Path $env:TEMP "animus-sync-$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

Write-Host "Starting license sync to ecosystem repositories..."
Write-Host "Temporary working directory: $tempDir"

foreach ($repo in $config.ecosystem) {
    Write-Host "`n----------------------------------------"
    Write-Host "Processing $($repo.name)..."
    
    $repoDir = Join-Path $tempDir $repo.name
    
    # Clone the repository
    Write-Host "Cloning $($repo.url)..."
    git clone $repo.url $repoDir
    
    if (-not (Test-Path $repoDir)) {
        Write-Error "Failed to clone $($repo.name). Skipping..."
        continue
    }
    
    # Copy the master license
    Write-Host "Copying master LICENSE to $($repo.name)..."
    Copy-Item -Path $masterLicense -Destination (Join-Path $repoDir "LICENSE") -Force
    
    # Commit and push
    Push-Location $repoDir
    
    $status = git status --porcelain
    if ($status -match "LICENSE") {
        Write-Host "Changes detected in LICENSE. Committing..."
        git add LICENSE
        git commit -m "chore: sync Animus Master License"
        
        Write-Host "Pushing to origin..."
        # If this is run in an automated environment, credentials must be cached/configured
        git push origin main
        Write-Host "Successfully synced $($repo.name)."
    } else {
        Write-Host "LICENSE is already up to date for $($repo.name)."
    }
    
    Pop-Location
}

Write-Host "`n----------------------------------------"
Write-Host "Sync complete. Cleaning up temporary directory..."
Remove-Item -Path $tempDir -Recurse -Force
Write-Host "Done!"
