-- Personality models table
CREATE TABLE IF NOT EXISTS personality_models (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  model_path TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'training', -- training, ready, failed
  traits JSONB,
  training_started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  training_completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on status for efficient queries
CREATE INDEX IF NOT EXISTS personality_models_status_idx ON personality_models(status);

-- Personality training sessions table
CREATE TABLE IF NOT EXISTS personality_training_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  content_ids UUID[] NOT NULL,
  conversation_ids UUID[] NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', -- pending, processing, completed, failed
  result JSONB,
  error TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for efficient queries
CREATE INDEX IF NOT EXISTS personality_training_sessions_user_id_idx ON personality_training_sessions(user_id);

-- Create index on status for efficient queries
CREATE INDEX IF NOT EXISTS personality_training_sessions_status_idx ON personality_training_sessions(status);

-- Personality response logs table
CREATE TABLE IF NOT EXISTS personality_response_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  input_message TEXT NOT NULL,
  output_message TEXT NOT NULL,
  traits_used JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for efficient queries
CREATE INDEX IF NOT EXISTS personality_response_logs_user_id_idx ON personality_response_logs(user_id);

-- Create index on conversation_id for efficient queries
CREATE INDEX IF NOT EXISTS personality_response_logs_conversation_id_idx ON personality_response_logs(conversation_id);