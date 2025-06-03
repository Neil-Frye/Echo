# AI Voice Service

The AI Voice Service for EthernalEcho provides voice synthesis capabilities to recreate a user's voice with high fidelity. It handles voice sample collection, processing, model training, and speech synthesis.

## Features

- Voice sample upload and processing
- Voice model training
- Text-to-speech synthesis with emotion control
- Voice profile management

## API Endpoints

### POST /api/v1/voice/upload

Upload voice samples for training.

**Request:**
- Headers: 
  - Authorization: Bearer {token}
- Body: FormData with 'samples' field containing audio files

**Response:**
```json
{
  "success": true,
  "data": {
    "sampleIds": ["uuid1", "uuid2", ...],
    "sampleCount": 10,
    "totalDuration": 600,
    "existingDuration": 1200,
    "combinedDuration": 1800,
    "message": "Voice samples uploaded successfully"
  }
}
```

### POST /api/v1/voice/train

Start voice model training.

**Request:**
- Headers: 
  - Authorization: Bearer {token}
  - Content-Type: application/json
- Body:
```json
{
  "sampleIds": ["uuid1", "uuid2", ...]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "jobId": "job-uuid",
    "status": "training_started",
    "message": "Voice model training has been initiated"
  }
}
```

### POST /api/v1/voice/synthesize

Synthesize speech from text.

**Request:**
- Headers: 
  - Authorization: Bearer {token}
  - Content-Type: application/json
- Body:
```json
{
  "text": "Hello, this is a test message.",
  "emotion": "happy" // Optional, defaults to "neutral"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "audioUrl": "https://bucket.s3.amazonaws.com/user-id/synthesized/audio-id.wav",
    "text": "Hello, this is a test message.",
    "emotion": "happy"
  }
}
```

### GET /api/v1/voice/profile

Get voice profile status.

**Request:**
- Headers: 
  - Authorization: Bearer {token}

**Response:**
```json
{
  "success": true,
  "data": {
    "exists": true,
    "profile": {
      "id": "profile-uuid",
      "userId": "user-uuid",
      "sampleCount": 30,
      "totalDurationSeconds": 1800,
      "qualityScore": 0.85,
      "isVerified": true,
      "trainingStatus": "completed",
      "modelVersion": "1.0",
      "voiceModelUrl": "https://bucket.s3.amazonaws.com/user-id/models/voice_model.bin",
      "createdAt": "2023-01-01T00:00:00Z",
      "updatedAt": "2023-01-02T00:00:00Z"
    }
  }
}
```

## Technical Details

### Voice Processing Pipeline

1. **Sample Collection**: User uploads at least 30 minutes of high-quality voice recordings
2. **Preprocessing**: Conversion to WAV format, normalization, noise reduction
3. **Feature Extraction**: Extract voice characteristics and unique vocal patterns
4. **Model Training**: Train a personalized voice synthesis model
5. **Evaluation**: Verify model quality with objective metrics
6. **Deployment**: Make model available for synthesis API

### Voice Model Architecture

The voice synthesis model uses a hybrid approach:
- Deep neural networks for voice characteristics extraction
- Statistical parametric models for voice reconstruction
- WaveNet-based vocoder for high-fidelity waveform generation

### Quality Requirements

- Minimum 30 minutes of voice samples required
- Multiple emotional tones and speech patterns
- Clear, noise-free recordings recommended
- Model quality score of at least 0.80 required for verification

## Development

### Prerequisites

- Node.js 16+
- PostgreSQL 13+
- AWS S3 bucket for sample storage
- FFmpeg for audio processing

### Setup

1. Install dependencies:
```
npm install
```

2. Create a `.env` file with the following:
```
PORT=3002
DATABASE_URL=postgres://user:password@localhost:5432/ethernalecho
JWT_SECRET=your-secret-key
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=ethernalecho-voice-samples
```

3. Run database migrations:
```
psql -U user -d ethernalecho -f src/db/migrations/voice-tables.sql
```

4. Start the development server:
```
npm run dev
```

### Testing

Run tests with:
```
npm test
```