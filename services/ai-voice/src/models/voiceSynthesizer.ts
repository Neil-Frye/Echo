import { S3 } from 'aws-sdk';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';
import { v4 as uuidv4 } from 'uuid';
import ffmpeg from 'fluent-ffmpeg';
import { TrainingStatus, VoiceProfile } from '@ethernalecho/shared';
import { db } from '../db';

const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const mkdir = promisify(fs.mkdir);
const exists = promisify(fs.exists);

export class VoiceSynthesizer {
  private s3: S3;
  private bucketName: string;
  private tempDir: string;

  constructor() {
    this.s3 = new S3({
      region: process.env.AWS_REGION || 'us-east-1',
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    });
    this.bucketName = process.env.S3_BUCKET_NAME || 'ethernalecho-voice-samples';
    this.tempDir = path.join(process.cwd(), 'temp');
    this.ensureTempDir();
  }

  private async ensureTempDir(): Promise<void> {
    if (!(await exists(this.tempDir))) {
      await mkdir(this.tempDir, { recursive: true });
    }
  }

  /**
   * Calculate the total duration of voice samples
   */
  public async calculateTotalDuration(files: Express.Multer.File[]): Promise<number> {
    let totalDuration = 0;

    for (const file of files) {
      try {
        const duration = await this.getAudioDuration(file.path);
        totalDuration += duration;
      } catch (error) {
        console.error(`Error calculating duration for file ${file.originalname}:`, error);
        // Continue with other files even if one fails
      }
    }

    return totalDuration;
  }

  /**
   * Get audio duration in seconds
   */
  private getAudioDuration(filePath: string): Promise<number> {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(filePath, (err, metadata) => {
        if (err) return reject(err);
        
        if (metadata && metadata.format && metadata.format.duration) {
          resolve(metadata.format.duration);
        } else {
          reject(new Error('Unable to determine audio duration'));
        }
      });
    });
  }

  /**
   * Process and store voice samples
   */
  public async processSamples(userId: string, files: Express.Multer.File[]): Promise<string[]> {
    const sampleIds: string[] = [];
    let totalDuration = 0;

    for (const file of files) {
      try {
        // Convert to WAV format if needed
        const wavPath = await this.convertToWav(file.path);
        
        // Calculate duration
        const duration = await this.getAudioDuration(wavPath);
        totalDuration += duration;
        
        // Upload to S3
        const sampleId = uuidv4();
        const s3Key = `${userId}/samples/${sampleId}.wav`;
        
        await this.s3.upload({
          Bucket: this.bucketName,
          Key: s3Key,
          Body: fs.createReadStream(wavPath),
          ContentType: 'audio/wav'
        }).promise();
        
        // Store metadata in database
        await db.query(
          `INSERT INTO voice_samples (id, user_id, s3_url, duration_seconds, original_filename)
           VALUES ($1, $2, $3, $4, $5)`,
          [
            sampleId,
            userId,
            `https://${this.bucketName}.s3.amazonaws.com/${s3Key}`,
            duration,
            file.originalname
          ]
        );
        
        sampleIds.push(sampleId);
        
        // Clean up temporary files
        fs.unlinkSync(wavPath);
        if (wavPath !== file.path) {
          fs.unlinkSync(file.path);
        }
      } catch (error) {
        console.error(`Error processing file ${file.originalname}:`, error);
        // Continue with other files even if one fails
      }
    }
    
    // Update or create voice profile
    const voiceProfileResult = await db.query(
      `SELECT * FROM voice_profiles WHERE user_id = $1`,
      [userId]
    );
    
    if (voiceProfileResult.rows.length > 0) {
      // Update existing profile
      await db.query(
        `UPDATE voice_profiles 
         SET sample_count = sample_count + $1,
             total_duration_seconds = total_duration_seconds + $2,
             updated_at = NOW()
         WHERE user_id = $3`,
        [files.length, totalDuration, userId]
      );
    } else {
      // Create new profile
      await db.query(
        `INSERT INTO voice_profiles 
         (id, user_id, sample_count, total_duration_seconds, training_status)
         VALUES ($1, $2, $3, $4, $5)`,
        [
          uuidv4(),
          userId,
          files.length,
          totalDuration,
          TrainingStatus.NotStarted
        ]
      );
    }
    
    return sampleIds;
  }

  /**
   * Convert audio file to WAV format
   */
  private async convertToWav(filePath: string): Promise<string> {
    const outputPath = path.join(this.tempDir, `${path.basename(filePath)}.wav`);
    
    return new Promise((resolve, reject) => {
      ffmpeg(filePath)
        .audioCodec('pcm_s16le')
        .audioChannels(1)
        .audioFrequency(16000)
        .format('wav')
        .on('end', () => resolve(outputPath))
        .on('error', (err) => reject(err))
        .save(outputPath);
    });
  }

  /**
   * Start voice model training
   */
  public async startTraining(userId: string, sampleIds: string[]): Promise<string> {
    // Check if enough samples exist
    const samplesResult = await db.query(
      `SELECT COUNT(*) as count, SUM(duration_seconds) as total_duration 
       FROM voice_samples WHERE user_id = $1`,
      [userId]
    );
    
    const { count, total_duration } = samplesResult.rows[0];
    
    if (count < 5 || total_duration < 300) { // At least 5 samples and 5 minutes
      throw new Error('Insufficient voice samples. Need at least 5 samples totaling 5 minutes.');
    }
    
    // Create training job
    const jobId = uuidv4();
    
    await db.query(
      `INSERT INTO training_jobs 
       (id, user_id, job_type, status, started_at)
       VALUES ($1, $2, $3, $4, NOW())`,
      [jobId, userId, 'voice', 'queued']
    );
    
    // Update voice profile status
    await db.query(
      `UPDATE voice_profiles 
       SET training_status = $1, updated_at = NOW()
       WHERE user_id = $2`,
      [TrainingStatus.InProgress, userId]
    );
    
    // In a real implementation, we would start a background process or queue a job
    // For now, we'll simulate training completion after a delay
    setTimeout(() => this.simulateTrainingCompletion(userId, jobId), 60000); // 1 minute
    
    return jobId;
  }

  /**
   * Simulate training completion (for demo purposes)
   * In a real implementation, this would be handled by a background worker
   */
  private async simulateTrainingCompletion(userId: string, jobId: string): Promise<void> {
    try {
      // Simulate model creation
      const modelUrl = `https://${this.bucketName}.s3.amazonaws.com/${userId}/models/voice_model_${Date.now()}.bin`;
      
      // Update training job
      await db.query(
        `UPDATE training_jobs 
         SET status = $1, completed_at = NOW()
         WHERE id = $2`,
        ['completed', jobId]
      );
      
      // Update voice profile
      await db.query(
        `UPDATE voice_profiles 
         SET training_status = $1, 
             voice_model_url = $2,
             is_verified = true,
             quality_score = $3,
             model_version = $4,
             updated_at = NOW()
         WHERE user_id = $5`,
        [
          TrainingStatus.Completed,
          modelUrl,
          0.85, // Mock quality score
          '1.0',
          userId
        ]
      );
      
      console.log(`Training completed for user ${userId}, job ${jobId}`);
    } catch (error) {
      console.error('Error in simulated training completion:', error);
      
      // Update job and profile to failed state
      await db.query(
        `UPDATE training_jobs SET status = 'failed', completed_at = NOW() WHERE id = $1`,
        [jobId]
      );
      
      await db.query(
        `UPDATE voice_profiles SET training_status = $1, updated_at = NOW() WHERE user_id = $2`,
        [TrainingStatus.Failed, userId]
      );
    }
  }

  /**
   * Synthesize speech from text
   */
  public async synthesize(userId: string, text: string, emotion: string = 'neutral'): Promise<string> {
    // Get user's voice profile
    const profileResult = await db.query(
      `SELECT * FROM voice_profiles WHERE user_id = $1`,
      [userId]
    );
    
    if (profileResult.rows.length === 0) {
      throw new Error('Voice profile not found');
    }
    
    const profile: VoiceProfile = profileResult.rows[0];
    
    if (profile.trainingStatus !== TrainingStatus.Completed) {
      throw new Error('Voice model training not completed');
    }
    
    // In a real implementation, we would use the trained model to generate speech
    // For now, we'll simulate with a placeholder URL
    
    const audioId = uuidv4();
    const audioUrl = `https://${this.bucketName}.s3.amazonaws.com/${userId}/synthesized/${audioId}.wav`;
    
    // Store synthesis record
    await db.query(
      `INSERT INTO synthesized_speech 
       (id, user_id, text, emotion, audio_url, created_at)
       VALUES ($1, $2, $3, $4, $5, NOW())`,
      [audioId, userId, text, emotion, audioUrl]
    );
    
    return audioUrl;
  }

  /**
   * Get voice profile for a user
   */
  public async getVoiceProfile(userId: string): Promise<VoiceProfile | null> {
    const result = await db.query(
      `SELECT * FROM voice_profiles WHERE user_id = $1`,
      [userId]
    );
    
    if (result.rows.length === 0) {
      return null;
    }
    
    return result.rows[0] as VoiceProfile;
  }
}