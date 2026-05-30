# TAAC 2026 Windows Runner
# Usage: .\run.ps1 [train|val|eval|infer] [args...]

param(
    [Parameter(Position=0)]
    [string]$Command = "train"
)

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$Env:TAAC_PROJECT_DIR = $ProjectDir

# Ensure PYTHONPATH includes src and experiments
$srcPath = Join-Path $ProjectDir "src"
$expPath = Join-Path $ProjectDir "experiments"

if ($Env:PYTHONPATH) {
    $Env:PYTHONPATH = "$srcPath;$expPath;$Env:PYTHONPATH"
} else {
    $Env:PYTHONPATH = "$srcPath;$expPath"
}

# Resolve python from venv if present, else fallback
$venvPython = Join-Path $ProjectDir "venv_taac\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

# Pass through to bootstrap module
$remainingArgs = $args
& $python -m taac2026.application.bootstrap.run_sh $Command @remainingArgs

exit $LASTEXITCODE
