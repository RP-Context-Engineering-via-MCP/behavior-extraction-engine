# ============================================================================
# Setup Profile Signals Table - PowerShell Script
# ============================================================================
# This script creates the user_profile_signals table in your PostgreSQL database
# 
# Prerequisites:
# - PostgreSQL installed with psql command available
# - Database connection details in .env file
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Profile Signals Table Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Load environment variables from .env file
if (Test-Path ".env") {
    Write-Host "Loading database configuration from .env..." -ForegroundColor Yellow
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
} else {
    Write-Host "Error: .env file not found!" -ForegroundColor Red
    exit 1
}

# Parse DATABASE_URL
$DATABASE_URL = $env:DATABASE_URL
if (-not $DATABASE_URL) {
    Write-Host "Error: DATABASE_URL not found in .env file!" -ForegroundColor Red
    exit 1
}

Write-Host "Database URL: $DATABASE_URL" -ForegroundColor Gray
Write-Host ""

# Extract connection details from DATABASE_URL
# Format: postgresql://username:password@host:port/database
if ($DATABASE_URL -match 'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)') {
    $DB_USER = $matches[1]
    $DB_PASSWORD = $matches[2]
    $DB_HOST = $matches[3]
    $DB_PORT = $matches[4]
    $DB_NAME = $matches[5]
    
    Write-Host "Connection Details:" -ForegroundColor Cyan
    Write-Host "  Host: $DB_HOST" -ForegroundColor Gray
    Write-Host "  Port: $DB_PORT" -ForegroundColor Gray
    Write-Host "  Database: $DB_NAME" -ForegroundColor Gray
    Write-Host "  User: $DB_USER" -ForegroundColor Gray
    Write-Host ""
    
    # Set PGPASSWORD environment variable to avoid password prompt
    $env:PGPASSWORD = $DB_PASSWORD
    
    # Run the SQL script
    Write-Host "Creating user_profile_signals table..." -ForegroundColor Yellow
    
    $sqlFile = "db\create_profile_signals_table.sql"
    
    if (Test-Path $sqlFile) {
        psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $sqlFile
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✓ Table created successfully!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "✗ Error creating table. Check logs above." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Error: SQL file not found at $sqlFile" -ForegroundColor Red
        exit 1
    }
    
    # Clear password from environment
    $env:PGPASSWORD = ""
    
} else {
    Write-Host "Error: Invalid DATABASE_URL format!" -ForegroundColor Red
    Write-Host "Expected format: postgresql://username:password@host:port/database" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
