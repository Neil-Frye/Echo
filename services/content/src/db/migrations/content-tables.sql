-- Content items table
CREATE TABLE IF NOT EXISTS content_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  content_type TEXT NOT NULL, -- mime type like 'image/jpeg', 'audio/mp3', etc.
  title TEXT NOT NULL,
  description TEXT,
  s3_url TEXT NOT NULL,
  file_size_bytes BIGINT NOT NULL,
  metadata JSONB,
  tags TEXT[] DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for efficient queries
CREATE INDEX IF NOT EXISTS content_items_user_id_idx ON content_items(user_id);

-- Create index on content_type for efficient queries
CREATE INDEX IF NOT EXISTS content_items_content_type_idx ON content_items(content_type);

-- Create index on tags for efficient queries (GIN index for array)
CREATE INDEX IF NOT EXISTS content_items_tags_idx ON content_items USING GIN(tags);

-- Create index on created_at for efficient sorting
CREATE INDEX IF NOT EXISTS content_items_created_at_idx ON content_items(created_at);

-- Content processing jobs table
CREATE TABLE IF NOT EXISTS content_processing_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content_id UUID NOT NULL REFERENCES content_items(id) ON DELETE CASCADE,
  job_type TEXT NOT NULL, -- 'ocr', 'speech-to-text', 'image-analysis', etc.
  status TEXT NOT NULL DEFAULT 'pending', -- pending, processing, completed, failed
  result JSONB,
  error TEXT,
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on content_id for efficient queries
CREATE INDEX IF NOT EXISTS content_processing_jobs_content_id_idx ON content_processing_jobs(content_id);

-- Create index on status for efficient queries
CREATE INDEX IF NOT EXISTS content_processing_jobs_status_idx ON content_processing_jobs(status);

-- Content collections table
CREATE TABLE IF NOT EXISTS content_collections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on user_id for efficient queries
CREATE INDEX IF NOT EXISTS content_collections_user_id_idx ON content_collections(user_id);

-- Content collection items table (junction table)
CREATE TABLE IF NOT EXISTS content_collection_items (
  collection_id UUID NOT NULL REFERENCES content_collections(id) ON DELETE CASCADE,
  content_id UUID NOT NULL REFERENCES content_items(id) ON DELETE CASCADE,
  added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  PRIMARY KEY(collection_id, content_id)
);

-- Create index on content_id for efficient queries
CREATE INDEX IF NOT EXISTS content_collection_items_content_id_idx ON content_collection_items(content_id);