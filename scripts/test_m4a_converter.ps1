# Test Script for M4A to MP3 Converter
# This script helps you test the converter with various scenarios

param(
    [switch]$CreateTestFile,
    [switch]$TestAll,
    [switch]$Cleanup
)

Write-Host "M4A to MP3 Converter Test Suite" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

# Function to create a test M4A file from existing audio
function Create-TestM4AFile {
    Write-Host "Creating test M4A file..." -ForegroundColor Yellow
    
    # Check if we have an existing audio file to convert
    $existingAudio = Get-ChildItem -Path "input" -Filter "*.mp3" | Select-Object -First 1
    
    if (-not $existingAudio) {
        Write-Host "No existing audio files found in input directory. Creating a simple test file..." -ForegroundColor Yellow
        
        # Create a simple test audio file using pydub
        $pythonCode = @"
import os
from pydub import AudioSegment
from pydub.generators import Sine

# Create a simple sine wave audio
audio = Sine(440).to_audio_segment(duration=5000)  # 5 seconds at 440Hz

# Save as M4A
output_path = "test_input.m4a"
audio.export(output_path, format="ipod")  # ipod format creates M4A
print(f"Created test file: {output_path}")
"@
        
        $pythonCode | python
    } else {
        Write-Host "Converting existing audio file to M4A for testing..." -ForegroundColor Yellow
        
        $pythonCode = @"
import os
from pydub import AudioSegment

# Load existing audio file
audio = AudioSegment.from_mp3("$($existingAudio.FullName)")

# Save as M4A
output_path = "test_input.m4a"
audio.export(output_path, format="ipod")
print(f"Created test file: {output_path}")
"@
        
        $pythonCode | python
    }
    
    if (Test-Path "test_input.m4a") {
        Write-Host "Test M4A file created successfully!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Failed to create test M4A file" -ForegroundColor Red
        return $false
    }
}

# Function to test basic conversion
function Test-BasicConversion {
    Write-Host "`nTesting basic conversion..." -ForegroundColor Yellow
    
    if (-not (Test-Path "test_input.m4a")) {
        Write-Host "No test M4A file found. Run with -CreateTestFile first." -ForegroundColor Red
        return $false
    }
    
    # Test basic conversion
    $result = & python convert_m4a_to_mp3.py "test_input.m4a"
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0 -and (Test-Path "test_input.mp3")) {
        Write-Host "✓ Basic conversion test passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Basic conversion test failed!" -ForegroundColor Red
        return $false
    }
}

# Function to test conversion with custom output
function Test-CustomOutput {
    Write-Host "`nTesting conversion with custom output..." -ForegroundColor Yellow
    
    if (-not (Test-Path "test_input.m4a")) {
        Write-Host "No test M4A file found. Run with -CreateTestFile first." -ForegroundColor Red
        return $false
    }
    
    # Test conversion with custom output
    $result = & python convert_m4a_to_mp3.py "test_input.m4a" "custom_output.mp3"
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0 -and (Test-Path "custom_output.mp3")) {
        Write-Host "✓ Custom output test passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Custom output test failed!" -ForegroundColor Red
        return $false
    }
}

# Function to test different bitrates
function Test-DifferentBitrates {
    Write-Host "`nTesting different bitrates..." -ForegroundColor Yellow
    
    if (-not (Test-Path "test_input.m4a")) {
        Write-Host "No test M4A file found. Run with -CreateTestFile first." -ForegroundColor Red
        return $false
    }
    
    $bitrates = @("128k", "192k", "320k")
    $successCount = 0
    
    foreach ($bitrate in $bitrates) {
        $outputFile = "test_output_$($bitrate.Replace('k', '')).mp3"
        $result = & python convert_m4a_to_mp3.py "test_input.m4a" $outputFile --bitrate $bitrate
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0 -and (Test-Path $outputFile)) {
            Write-Host "✓ Bitrate $bitrate test passed!" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "✗ Bitrate $bitrate test failed!" -ForegroundColor Red
        }
    }
    
    return $successCount -eq $bitrates.Count
}

# Function to test PowerShell wrapper
function Test-PowerShellWrapper {
    Write-Host "`nTesting PowerShell wrapper..." -ForegroundColor Yellow
    
    if (-not (Test-Path "test_input.m4a")) {
        Write-Host "No test M4A file found. Run with -CreateTestFile first." -ForegroundColor Red
        return $false
    }
    
    # Test PowerShell wrapper
    $result = & .\convert_m4a_to_mp3.ps1 "test_input.m4a" "ps_output.mp3"
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0 -and (Test-Path "ps_output.mp3")) {
        Write-Host "✓ PowerShell wrapper test passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ PowerShell wrapper test failed!" -ForegroundColor Red
        return $false
    }
}

# Function to test batch file wrapper
function Test-BatchWrapper {
    Write-Host "`nTesting batch file wrapper..." -ForegroundColor Yellow
    
    if (-not (Test-Path "test_input.m4a")) {
        Write-Host "No test M4A file found. Run with -CreateTestFile first." -ForegroundColor Red
        return $false
    }
    
    # Test batch file wrapper
    $result = & .\convert_m4a_to_mp3.bat "test_input.m4a" "batch_output.mp3"
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0 -and (Test-Path "batch_output.mp3")) {
        Write-Host "✓ Batch file wrapper test passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Batch file wrapper test failed!" -ForegroundColor Red
        return $false
    }
}

# Function to test error handling
function Test-ErrorHandling {
    Write-Host "`nTesting error handling..." -ForegroundColor Yellow
    
    # Test with non-existent file
    $result = & python convert_m4a_to_mp3.py "nonexistent.m4a" 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -ne 0) {
        Write-Host "✓ Error handling test passed (correctly failed for non-existent file)!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Error handling test failed (should have failed for non-existent file)!" -ForegroundColor Red
        return $false
    }
}

# Function to cleanup test files
function Remove-TestFiles {
    Write-Host "`nCleaning up test files..." -ForegroundColor Yellow
    
    $testFiles = @(
        "test_input.m4a",
        "test_input.mp3",
        "custom_output.mp3",
        "test_output_128.mp3",
        "test_output_192.mp3",
        "test_output_320.mp3",
        "ps_output.mp3",
        "batch_output.mp3"
    )
    
    $removedCount = 0
    foreach ($file in $testFiles) {
        if (Test-Path $file) {
            Remove-Item $file -Force
            Write-Host "Removed: $file" -ForegroundColor Gray
            $removedCount++
        }
    }
    
    Write-Host "Cleaned up $removedCount test files" -ForegroundColor Green
}

# Main execution
if ($CreateTestFile) {
    Create-TestM4AFile
    exit
}

if ($Cleanup) {
    Remove-TestFiles
    exit
}

if ($TestAll) {
    Write-Host "Running all tests..." -ForegroundColor Cyan
    
    # Create test file if it doesn't exist
    if (-not (Test-Path "test_input.m4a")) {
        if (-not (Create-TestM4AFile)) {
            Write-Host "Failed to create test file. Exiting." -ForegroundColor Red
            exit 1
        }
    }
    
    $tests = @(
        @{ Name = "Basic Conversion"; Function = "Test-BasicConversion" },
        @{ Name = "Custom Output"; Function = "Test-CustomOutput" },
        @{ Name = "Different Bitrates"; Function = "Test-DifferentBitrates" },
        @{ Name = "PowerShell Wrapper"; Function = "Test-PowerShellWrapper" },
        @{ Name = "Batch File Wrapper"; Function = "Test-BatchWrapper" },
        @{ Name = "Error Handling"; Function = "Test-ErrorHandling" }
    )
    
    $passedTests = 0
    $totalTests = $tests.Count
    
    foreach ($test in $tests) {
        Write-Host "`nRunning: $($test.Name)" -ForegroundColor Cyan
        $result = & $test.Function
        if ($result) {
            $passedTests++
        }
    }
    
    Write-Host "`n" + "="*50 -ForegroundColor Cyan
    Write-Host "Test Results: $passedTests/$totalTests tests passed" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })
    Write-Host "="*50 -ForegroundColor Cyan
    
    if ($passedTests -eq $totalTests) {
        Write-Host "All tests passed! Your M4A converter is working correctly." -ForegroundColor Green
    } else {
        Write-Host "Some tests failed. Check the output above for details." -ForegroundColor Yellow
    }
} else {
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\test_m4a_converter.ps1 -CreateTestFile  # Create a test M4A file" -ForegroundColor White
    Write-Host "  .\test_m4a_converter.ps1 -TestAll          # Run all tests" -ForegroundColor White
    Write-Host "  .\test_m4a_converter.ps1 -Cleanup          # Remove test files" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run individual tests by calling the functions directly." -ForegroundColor Gray
} 