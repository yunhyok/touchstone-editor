[CmdletBinding()]
param([string]$IsccPath)

$ErrorActionPreference = 'Stop'
if (-not $IsccPath) {
    $IsccPath = @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $IsccPath) {
    $IsccPath = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
}
if (-not $IsccPath -or -not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) {
    throw 'Install Inno Setup 6 or pass its ISCC.exe path using -IsccPath.'
}
$IsccPath = (Resolve-Path -LiteralPath $IsccPath).Path
$pythonPath = (Get-Command python).Source
$pythonDirectory = Split-Path -Parent $pythonPath
$previousBuildPath = $env:PATH
# Prevent unrelated applications' ICU/OpenSSL DLLs from entering the bundle.
$env:PATH = "$pythonDirectory;$pythonDirectory\Scripts;$env:SystemRoot\System32;$env:SystemRoot"

Push-Location -LiteralPath (Join-Path $PSScriptRoot '..')
try {
    $version = python -c "import sys, sysconfig, tomllib; assert sysconfig.get_platform() == 'win-amd64' and sys.version_info[:2] == (3, 12), 'Build with CPython 3.12 x64'; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
    if ($LASTEXITCODE -ne 0 -or $version -notmatch '^\d+\.\d+\.\d+$') {
        throw 'Unable to read the release version from pyproject.toml.'
    }
    if ($env:GITHUB_REF_TYPE -eq 'tag' -and $env:GITHUB_REF_NAME -ne "v$version") {
        throw "Release tag must match v$version from pyproject.toml."
    }

    $assetsPath = (Resolve-Path -LiteralPath assets).Path
    python -m PyInstaller --noconfirm --clean --noupx --onedir --windowed `
        --name TouchstoneEditor --icon "$assetsPath/icon.ico" --add-data "$assetsPath;assets" `
        --specpath build --workpath build/pyinstaller run_app.py
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller failed.'
    }

    $bundle = 'dist/TouchstoneEditor'
    # Unused optional PDF and GPL-only virtual-keyboard plugins pull in these libraries.
    foreach ($relative in @(
        'plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll',
        'plugins/imageformats/qpdf.dll', 'Qt6VirtualKeyboard.dll', 'Qt6Pdf.dll',
        'Qt6Qml.dll', 'Qt6QmlMeta.dll', 'Qt6QmlModels.dll',
        'Qt6QmlWorkerScript.dll', 'Qt6Quick.dll'
    )) {
        $optionalFile = Join-Path "$bundle/_internal/PySide6" $relative
        if (Test-Path -LiteralPath $optionalFile) {
            Remove-Item -LiteralPath $optionalFile
        }
    }

    $smoke = Start-Process -FilePath (Resolve-Path 'dist/TouchstoneEditor/TouchstoneEditor.exe').Path `
        -ArgumentList '--smoke-test' -WindowStyle Hidden -PassThru
    if (-not $smoke.WaitForExit(20000)) {
        Stop-Process -Id $smoke.Id -ErrorAction SilentlyContinue
        throw 'Packaged app did not complete its startup smoke test.'
    }
    if ($smoke.ExitCode -ne 0) {
        throw "Packaged app startup failed with exit code $($smoke.ExitCode)."
    }

    Copy-Item -LiteralPath README.md, LICENSE, THIRD_PARTY_NOTICES.md -Destination $bundle
    Copy-Item -LiteralPath packaging/licenses -Destination $bundle -Recurse
    @'
import shutil
import sys
from importlib.metadata import files
from pathlib import Path

licenses = Path('dist/TouchstoneEditor/licenses')
shutil.copyfile(Path(sys.base_prefix) / 'LICENSE.txt', licenses / 'CPython.txt')
pyinstaller_license = next(path for path in files('PyInstaller') if path.name == 'COPYING.txt')
shutil.copyfile(pyinstaller_license.locate(), licenses / 'PyInstaller.txt')
'@ | python -
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to copy the runtime licenses.'
    }

    & $IsccPath "/DAppVersion=$version" 'packaging/TouchstoneEditor.iss'
    if ($LASTEXITCODE -ne 0) {
        throw 'Inno Setup failed.'
    }
    Write-Output "Installer: dist/Touchstone-Editor-Setup-v$version-x64.exe"
} finally {
    $env:PATH = $previousBuildPath
    Pop-Location
}
