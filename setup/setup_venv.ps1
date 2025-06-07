if (-Not (Test-Path ".venv")) {
  $pyVersions = @("python3.12", "python3.11", "python3.10", "python3.9", "python")
  $found = $false
  foreach ($py in $pyVersions) {
    try {
      $ver = & $py -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
      if ($ver -and [version]$ver -le [version]"3.12") {
        Write-Host "Using Python $ver for venv creation"
        & $py -m venv .venv
        & .\.venv\Scripts\pip.exe install --upgrade pip
        & .\.venv\Scripts\pip.exe install -r requirements.txt
        $found = $true
        break
      }
    } catch {
      # ignorieren, wenn python nicht gefunden
    }
  }
  if (-not $found) {
    Write-Error "No suitable Python version (<= 3.12) found!"
    exit 1
  }
} else {
  Write-Host "venv already exists"
}