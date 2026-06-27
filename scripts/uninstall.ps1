$ErrorActionPreference = "Stop"

$installDir = if ($env:WERK_INSTALL_DIR) {
    $env:WERK_INSTALL_DIR
} else {
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is not set; set WERK_INSTALL_DIR to choose an install directory"
    }

    Join-Path $env:LOCALAPPDATA "Programs\Werk1112\bin"
}

$binaryPath = Join-Path $installDir "werk.exe"
$modelStoreKept = $false

if (Test-Path $binaryPath) {
    Remove-Item -Path $binaryPath -Force
    Write-Host "Removed $binaryPath"
} else {
    Write-Host "Werk1112 is not installed."
}

$modelStore = $null

if ($env:WERK_HOME -and (Test-Path $env:WERK_HOME -PathType Container)) {
    $modelStore = $env:WERK_HOME
} elseif ($env:LOCALAPPDATA) {
    $localModelStore = Join-Path $env:LOCALAPPDATA "werk1112"
    if (Test-Path $localModelStore -PathType Container) {
        $modelStore = $localModelStore
    }
}

if ($modelStore) {
    Write-Host ""
    Write-Host "Werk1112 model store detected:"
    Write-Host ""
    Write-Host $modelStore
    Write-Host ""
    Write-Host "This directory may contain downloaded models."
    Write-Host ""

    $answer = Read-Host "Remove the model store? [y/N]"

    if ($answer -eq "y" -or $answer -eq "yes") {
        Remove-Item -Path $modelStore -Recurse -Force
    } else {
        $modelStoreKept = $true
    }
}

Write-Host ""
Write-Host "Werk1112 successfully removed."

if ($modelStoreKept) {
    Write-Host "Model store kept."
}
