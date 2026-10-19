#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

Set-Location $RepoRoot

$IsNativeWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
if (-not $IsNativeWindows) {
    throw "scripts/build-windows.ps1 must be run on native Windows."
}

# Prefer the Python launcher; accept standalone Python installations as well.
$PythonLauncher = $null
$PythonArgs = @()
foreach ($candidate in @("py", "python", "python3")) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command) {
        $PythonLauncher = $command.Source
        if ($candidate -eq "py") { $PythonArgs = @("-3") }
        break
    }
}
if (-not $PythonLauncher) {
    throw "Python 3.9+ is required to regenerate the CLI reference (py, python or python3 on PATH)."
}
& $PythonLauncher @PythonArgs -c 'import sys; sys.exit(sys.version_info < (3, 9))'
if ($LASTEXITCODE -ne 0) {
    throw "CLI reference generation requires a working Python 3.9+ installation."
}

function Prepend-EnvFlag {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Flag,

        [Parameter(Mandatory = $true)]
        [string]$Pattern
    )

    $current = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $current) {
        [Environment]::SetEnvironmentVariable($Name, $Flag, "Process")
    } elseif ($current -notmatch $Pattern) {
        [Environment]::SetEnvironmentVariable($Name, "$Flag $current", "Process")
    }
}

Prepend-EnvFlag `
    -Name "CL" `
    -Flag "/Zc:preprocessor" `
    -Pattern "/Zc:preprocessor"

Prepend-EnvFlag `
    -Name "NVCC_PREPEND_FLAGS" `
    -Flag "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING" `
    -Pattern "CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe not found. Run this from x64 Developer PowerShell/Native Tools for Visual Studio with the C++ build tools installed."
}

if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw "nvcc.exe not found. Install the CUDA Toolkit and make sure CUDA bin is on PATH."
}

$clPath = (Get-Command cl.exe).Source
if ($clPath -notmatch "\\Hostx64\\x64\\cl\.exe$") {
    throw "MSVC is not in x64 mode. Re-run Developer PowerShell with -arch=x64 -host_arch=x64, or open the x64 Native Tools shell."
}

Write-Host "Running Windows release build with CUDA/MSVC environment fixes."
Write-Host "  CL=$env:CL"
Write-Host "  NVCC_PREPEND_FLAGS=$env:NVCC_PREPEND_FLAGS"

cargo build --release --locked --no-default-features --target x86_64-pc-windows-msvc --features release-windows

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Validating build-specific CLI reference generation."
$BinaryPath = Join-Path $RepoRoot "target\x86_64-pc-windows-msvc\release\werk.exe"
$ReferenceDir = Join-Path ([System.IO.Path]::GetTempPath()) ("werk-cli-reference-" + [guid]::NewGuid().ToString("N"))
New-Item $ReferenceDir -ItemType Directory | Out-Null
try {
    & $PythonLauncher @PythonArgs (Join-Path $ScriptDir "generate-cli-reference.py") $BinaryPath --target x86_64-pc-windows-msvc --profile discovery --output-dir $ReferenceDir
    $ReferenceExitCode = $LASTEXITCODE
} finally {
    Remove-Item $ReferenceDir -Recurse -Force
}
if ($ReferenceExitCode -ne 0) {
    exit $ReferenceExitCode
}
