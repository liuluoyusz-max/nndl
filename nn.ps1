# Run this project with the user's existing NN environment, from any directory.
$nnPython = Join-Path $env:USERPROFILE '.conda\envs\NN\python.exe'
if (-not (Test-Path -LiteralPath $nnPython)) {
    throw "NN interpreter not found: $nnPython"
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $nnPython @args
    $nnExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $nnExitCode
