#!/bin/bash

# Run database migrations

echo "Running database migrations..."

# Create database if it doesn't exist
psql -U postgres -c "CREATE DATABASE ethernalecho;" 2>/dev/null || true

# Run migrations for each service
psql -U postgres -d ethernalecho -f ../services/auth/src/db/migrations/auth-tables.sql
psql -U postgres -d ethernalecho -f ../services/ai-voice/src/db/migrations/voice-tables.sql
psql -U postgres -d ethernalecho -f ../services/content/src/db/migrations/content-tables.sql
psql -U postgres -d ethernalecho -f ../services/ai-personality/src/db/migrations/personality-tables.sql
psql -U postgres -d ethernalecho -f ../services/conversation/src/db/migrations/conversation-tables.sql
psql -U postgres -d ethernalecho -f ../services/billing/src/db/migrations/billing-tables.sql

echo "Migrations completed!"