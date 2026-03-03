#!/bin/bash
# ============================================================================
# Setup Profile Signals Table - Bash Script
# ============================================================================
# This script creates the user_profile_signals table in your PostgreSQL database
# 
# Prerequisites:
# - PostgreSQL installed with psql command available
# - Database connection details in .env file
#
# Usage:
#   chmod +x setup_profile_signals_table.sh
#   ./setup_profile_signals_table.sh
# ============================================================================

echo "========================================"
echo "Profile Signals Table Setup"
echo "========================================"
echo ""

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading database configuration from .env..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

# Check if DATABASE_URL exists
if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL not found in .env file!"
    exit 1
fi

echo "Database URL: $DATABASE_URL"
echo ""

# Extract connection details from DATABASE_URL
# Format: postgresql://username:password@host:port/database
if [[ $DATABASE_URL =~ postgresql://([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+) ]]; then
    DB_USER="${BASH_REMATCH[1]}"
    DB_PASSWORD="${BASH_REMATCH[2]}"
    DB_HOST="${BASH_REMATCH[3]}"
    DB_PORT="${BASH_REMATCH[4]}"
    DB_NAME="${BASH_REMATCH[5]}"
    
    echo "Connection Details:"
    echo "  Host: $DB_HOST"
    echo "  Port: $DB_PORT"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo ""
    
    # Set PGPASSWORD environment variable to avoid password prompt
    export PGPASSWORD="$DB_PASSWORD"
    
    # Run the SQL script
    echo "Creating user_profile_signals table..."
    
    SQL_FILE="db/create_profile_signals_table.sql"
    
    if [ -f "$SQL_FILE" ]; then
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$SQL_FILE"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Table created successfully!"
        else
            echo ""
            echo "✗ Error creating table. Check logs above."
            exit 1
        fi
    else
        echo "Error: SQL file not found at $SQL_FILE"
        exit 1
    fi
    
    # Clear password from environment
    unset PGPASSWORD
    
else
    echo "Error: Invalid DATABASE_URL format!"
    echo "Expected format: postgresql://username:password@host:port/database"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
