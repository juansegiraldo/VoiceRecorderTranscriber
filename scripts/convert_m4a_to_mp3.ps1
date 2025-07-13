# M4A to MP3 Converter PowerShell Wrapper
# This script provides a convenient way to convert M4A files to MP3 on Windows

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$InputPath,
    
    [Parameter(Mandatory=$false, Position=1)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Bitrate = "192k",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Function to check if Python is available
function Test-PythonAvailable {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Python found: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Python not found in PATH" -ForegroundColor Red
        return $false
    }
    return $false
}

# Function to check if required packages are installed
function Test-RequiredPackages {
    try {
        $result = python -c "import pydub; print('pydub available')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Required packages are available" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Required packages not found. Installing..." -ForegroundColor Yellow
        return $false
    }
    return $false
}

# Function to install required packages
function Install-RequiredPackages {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    try {
        pip install pydub
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Packages installed successfully" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "Failed to install packages" -ForegroundColor Red
        return $false
    }
    return $false
}

# Main execution
Write-Host "M4A to MP3 Converter" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan

# Check if Python is available
if (-not (Test-PythonAvailable)) {
    Write-Host "Error: Python is not available. Please install Python and try again." -ForegroundColor Red
    exit 1
}

# Check if required packages are installed
if (-not (Test-RequiredPackages)) {
    if (-not (Install-RequiredPackages)) {
        Write-Host "Error: Failed to install required packages." -ForegroundColor Red
        exit 1
    }
}

# Build the Python command
$pythonScript = "convert_m4a_to_mp3.py"
$pythonArgs = @($InputPath)

if ($OutputPath) {
    $pythonArgs += $OutputPath
}

if ($Bitrate -ne "192k") {
    $pythonArgs += "--bitrate", $Bitrate
}

if ($Verbose) {
    $pythonArgs += "--verbose"
}

# Execute the Python script
Write-Host "Running conversion..." -ForegroundColor Yellow
Write-Host "Command: python $pythonScript $($pythonArgs -join ' ')" -ForegroundColor Gray

try {
    & python $pythonScript @pythonArgs
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host "Conversion completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Conversion failed with exit code: $exitCode" -ForegroundColor Red
    }
    
    exit $exitCode
}
catch {
    Write-Host "Error executing Python script: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 