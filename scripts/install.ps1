$ErrorActionPreference = "Stop"

function Get-WerkVersion {
    param([string]$InputVersion)

    if ($InputVersion -eq "latest") {
        $latest = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/$script:WerkRepo/releases/latest" `
            -Headers @{ "User-Agent" = "werk1112-installer" }

        if (-not $latest.tag_name) {
            throw "Could not resolve latest release for $script:WerkRepo"
        }

        $InputVersion = [string]$latest.tag_name
    }

    if ($InputVersion.StartsWith("v")) {
        return @{
            Tag = $InputVersion
            Version = $InputVersion.Substring(1)
        }
    }

    return @{
        Tag = "v$InputVersion"
        Version = $InputVersion
    }
}

function Test-PathContainsEntry {
    param(
        [string]$PathValue,
        [string]$Entry
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $false
    }

    $normalizedEntry = $Entry.TrimEnd('\')
    foreach ($pathEntry in ($PathValue -split ';')) {
        if ([string]::IsNullOrWhiteSpace($pathEntry)) {
            continue
        }

        if ([string]::Equals($pathEntry.TrimEnd('\'), $normalizedEntry, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }

    return $false
}

$script:WerkRepo = if ($env:WERK_REPO) { $env:WERK_REPO } else { "philipbodenbach/werk1112" }
$versionInput = if ($env:WERK_VERSION) { $env:WERK_VERSION } else { "latest" }
$installDir = if ($env:WERK_INSTALL_DIR) {
    $env:WERK_INSTALL_DIR
} else {
    if (-not $env:LOCALAPPDATA) {
        throw "LOCALAPPDATA is not set; set WERK_INSTALL_DIR to choose an install directory"
    }

    Join-Path $env:LOCALAPPDATA "Programs\Werk1112\bin"
}

$version = Get-WerkVersion -InputVersion $versionInput
$artifactName = "werk1112-v$($version.Version)-windows-x86_64.zip"
$downloadUrl = "https://github.com/$script:WerkRepo/releases/download/$($version.Tag)/$artifactName"

$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("werk1112-install-" + [System.Guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $tempDir $artifactName

try {
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

    Write-Host "Downloading $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -UseBasicParsing

    Expand-Archive -Path $archivePath -DestinationPath $tempDir -Force

    $extractedBinary = Join-Path $tempDir "werk.exe"
    if (-not (Test-Path $extractedBinary)) {
        throw "Downloaded artifact did not contain werk.exe"
    }

    New-Item -ItemType Directory -Path $installDir -Force | Out-Null

    $installedBinary = Join-Path $installDir "werk.exe"
    Copy-Item -Path $extractedBinary -Destination $installedBinary -Force

    Write-Host "Installed $installedBinary"

    if (-not (Test-PathContainsEntry -PathValue $env:Path -Entry $installDir)) {
        Write-Warning "$installDir is not on PATH. Add it to PATH to run werk from any directory."
    }

    if ($env:WERK_ADD_TO_PATH -eq "1") {
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

        if (-not (Test-PathContainsEntry -PathValue $userPath -Entry $installDir)) {
            $newUserPath = if ([string]::IsNullOrWhiteSpace($userPath)) {
                $installDir
            } else {
                "$userPath;$installDir"
            }

            [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
            Write-Host "Added $installDir to the user PATH. Open a new terminal before running werk."
        } else {
            Write-Host "$installDir is already in the user PATH."
        }
    }

    Write-Host ""
    Write-Host "Werk1112 installed successfully."
    Write-Host ""
    Write-Host "Run:"
    Write-Host "  werk --help"
} finally {
    if (Test-Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force
    }
}
