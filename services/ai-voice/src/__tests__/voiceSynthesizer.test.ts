import { VoiceSynthesizer } from '../models/voiceSynthesizer';
import { TrainingStatus } from '@ethernalecho/shared';

// Mock the database
jest.mock('../db', () => ({
  db: {
    query: jest.fn().mockImplementation((query, params) => {
      if (query.includes('SELECT * FROM voice_profiles')) {
        return {
          rows: [
            {
              id: 'test-profile-id',
              userId: 'test-user-id',
              sampleCount: 10,
              totalDurationSeconds: 600,
              trainingStatus: TrainingStatus.Completed,
              voiceModelUrl: 'https://test-bucket.s3.amazonaws.com/test-user-id/models/voice_model.bin',
              isVerified: true,
              qualityScore: 0.85,
              modelVersion: '1.0',
              createdAt: new Date(),
              updatedAt: new Date()
            }
          ]
        };
      }
      
      if (query.includes('INSERT INTO')) {
        return { rows: [{ id: 'test-id' }] };
      }
      
      if (query.includes('UPDATE')) {
        return { rows: [] };
      }
      
      if (query.includes('SELECT COUNT')) {
        return { 
          rows: [
            { count: 10, total_duration: 600 }
          ]
        };
      }
      
      return { rows: [] };
    })
  }
}));

// Mock AWS S3
jest.mock('aws-sdk', () => {
  return {
    S3: jest.fn().mockImplementation(() => {
      return {
        upload: jest.fn().mockReturnValue({
          promise: jest.fn().mockResolvedValue({})
        })
      };
    })
  };
});

// Mock fluent-ffmpeg
jest.mock('fluent-ffmpeg', () => {
  return jest.fn().mockImplementation(() => {
    return {
      audioCodec: jest.fn().mockReturnThis(),
      audioChannels: jest.fn().mockReturnThis(),
      audioFrequency: jest.fn().mockReturnThis(),
      format: jest.fn().mockReturnThis(),
      on: jest.fn().mockImplementation((event, callback) => {
        if (event === 'end') {
          callback();
        }
        return this;
      }),
      save: jest.fn()
    };
  });
});

// Mock fs functions
jest.mock('fs', () => {
  return {
    ...jest.requireActual('fs'),
    createReadStream: jest.fn(),
    unlinkSync: jest.fn(),
    exists: jest.fn().mockImplementation((path, callback) => {
      callback(null, true);
    })
  };
});

describe('VoiceSynthesizer', () => {
  let synthesizer: VoiceSynthesizer;
  
  beforeEach(() => {
    synthesizer = new VoiceSynthesizer();
    
    // Mock the private methods
    (synthesizer as any).getAudioDuration = jest.fn().mockResolvedValue(60); // 60 seconds
    (synthesizer as any).convertToWav = jest.fn().mockResolvedValue('/tmp/test.wav');
  });
  
  afterEach(() => {
    jest.clearAllMocks();
  });
  
  it('should calculate total duration of audio files', async () => {
    const files = [
      { path: '/tmp/file1.mp3', originalname: 'file1.mp3' },
      { path: '/tmp/file2.wav', originalname: 'file2.wav' }
    ] as Express.Multer.File[];
    
    const totalDuration = await synthesizer.calculateTotalDuration(files);
    
    // Each file is mocked to be 60 seconds, so total should be 120
    expect(totalDuration).toBe(120);
    expect((synthesizer as any).getAudioDuration).toHaveBeenCalledTimes(2);
  });
  
  it('should process and store voice samples', async () => {
    const files = [
      { path: '/tmp/file1.mp3', originalname: 'file1.mp3' },
      { path: '/tmp/file2.wav', originalname: 'file2.wav' }
    ] as Express.Multer.File[];
    
    const sampleIds = await synthesizer.processSamples('test-user-id', files);
    
    expect(sampleIds).toHaveLength(2);
    expect((synthesizer as any).convertToWav).toHaveBeenCalledTimes(2);
  });
  
  it('should start voice model training', async () => {
    const jobId = await synthesizer.startTraining('test-user-id', ['sample1', 'sample2']);
    
    expect(typeof jobId).toBe('string');
    expect(jobId).toHaveLength(36); // UUID length
  });
  
  it('should synthesize speech from text', async () => {
    const audioUrl = await synthesizer.synthesize('test-user-id', 'Hello, this is a test');
    
    expect(audioUrl).toContain('https://');
    expect(audioUrl).toContain('.wav');
  });
  
  it('should get voice profile for a user', async () => {
    const profile = await synthesizer.getVoiceProfile('test-user-id');
    
    expect(profile).toBeDefined();
    expect(profile?.userId).toBe('test-user-id');
    expect(profile?.trainingStatus).toBe(TrainingStatus.Completed);
  });
});