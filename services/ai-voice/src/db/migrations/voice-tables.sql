-- Voice samples table
CREATE TABLE IF NOT EXISTS voice_samples (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  s3_url TEXT NOT NULL,
  duration_seconds FLOAT NOT NULL,
  original_filename TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Voice profiles table
CREATE TABLE IF NOT EXISTS voice_profiles (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE,
  voice_model_url TEXT,
  sample_count INTEGER NOT NULL DEFAULT 0,
  total_duration_seconds FLOAT NOT NULL DEFAULT 0,
  quality_score FLOAT,
  is_verified BOOLEAN NOT NULL DEFAULT FALSE,
  training_status TEXT NOT NULL,
  model_version TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Voice training jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE NOT NULL,
  completed_at TIMESTAMP WITH TIME ZONE,
  error_message TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Synthesized speech table
CREATE TABLE IF NOT EXISTS synthesized_speech (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  text TEXT NOT NULL,
  emotion TEXT NOT NULL DEFAULT 'neutral',
  audio_url TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_voice_samples_user_id ON voice_samples(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_synthesized_speech_user_id ON synthesized_speech(user_id);