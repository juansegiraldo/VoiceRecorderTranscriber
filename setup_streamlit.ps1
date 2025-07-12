# Voice Transcriber Streamlit App Setup Script
Write-Host "🎤 Voice Transcriber Streamlit App Setup" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = py --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    try {
        $pythonVersion = python --version
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
        Write-Host "You can download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
}

# Check if pip is available
Write-Host "Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = py -m pip --version
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
    $usePy = $true
} catch {
    try {
        $pipVersion = pip --version
        Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
        $usePy = $false
    } catch {
        Write-Host "❌ pip not found. Please install pip first." -ForegroundColor Red
        exit 1
    }
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
if ($usePy) {
    py -m pip install -r requirements.txt
} else {
    pip install -r requirements.txt
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to install dependencies." -ForegroundColor Red
    Write-Host "Try running manually: py -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    @"
# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_api_key_here
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ .env file created. Please add your OpenAI API key." -ForegroundColor Green
} else {
    Write-Host "✅ .env file already exists." -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the app:" -ForegroundColor Cyan
Write-Host "1. Add your OpenAI API key to the .env file" -ForegroundColor White
Write-Host "2. Run: py -m streamlit run AppTranscribe.py" -ForegroundColor White
Write-Host ""
Write-Host "The app will open in your browser automatically." -ForegroundColor Cyan 