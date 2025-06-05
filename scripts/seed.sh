#!/bin/bash

# Run database seed scripts

echo "Seeding database..."

# Create test user accounts
psql -U postgres -d ethernalecho -c "
INSERT INTO users (email, password_hash, full_name, email_verified, status)
VALUES 
('demo@example.com', '\$2b\$10\$EExj8MuFG1HOvIs3YJsA9O1C5Jn2G3pK9s5RFpiXgA95w.NRxNrte', 'Demo User', TRUE, 'active'),
('admin@ethernalecho.com', '\$2b\$10\$YaB29JfbIQP5Sznac1CFTuoH3nBDXaiNK9JEw1SXsKjK3UOKhXHVa', 'Admin User', TRUE, 'active')
ON CONFLICT (email) DO NOTHING;
"

# Create subscription plans
psql -U postgres -d ethernalecho -c "
INSERT INTO subscription_plans (name, price_monthly, price_yearly, features, storage_limit, family_member_limit)
VALUES 
('Basic', 999, 9999, '{\"voice_samples\":50,\"ai_responses\":100,\"conversations\":10}', 5, 2),
('Premium', 1999, 19999, '{\"voice_samples\":100,\"ai_responses\":500,\"conversations\":50}', 50, 5),
('Family', 2999, 29999, '{\"voice_samples\":250,\"ai_responses\":1000,\"conversations\":100}', 100, 10)
ON CONFLICT (name) DO NOTHING;
"

echo "Seeding completed!"