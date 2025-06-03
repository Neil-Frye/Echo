import { Request, Response, NextFunction } from 'express';
import { VoiceSynthesizer } from '../models/voiceSynthesizer';
import { ApiError } from '../middleware/errorHandler';

const synthesizer = new VoiceSynthesizer();

/**
 * Upload voice samples for training
 */
export const uploadSamples = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user.id;
    const files = req.files as Express.Multer.File[];
    
    if (!files || files.length === 0) {
      throw new ApiError(400, 'No files uploaded');
    }
    
    // Calculate total duration
    const totalDuration = await synthesizer.calculateTotalDuration(files);
    
    // Validate minimum duration (30 minutes = 1800 seconds)
    const existingProfile = await synthesizer.getVoiceProfile(userId);
    const existingDuration = existingProfile?.totalDurationSeconds || 0;
    
    if (totalDuration + existingDuration < 1800) {
      return res.status(200).json({
        success: true,
        data: {
          message: 'Samples uploaded but more are needed',
          sampleCount: files.length,
          totalDuration,
          existingDuration,
          combinedDuration: totalDuration + existingDuration,
          requiredDuration: 1800,
          remainingDuration: 1800 - (totalDuration + existingDuration)
        }
      });
    }
    
    // Process and store samples
    const sampleIds = await synthesizer.processSamples(userId, files);
    
    res.status(200).json({
      success: true,
      data: {
        sampleIds,
        sampleCount: files.length,
        totalDuration,
        existingDuration,
        combinedDuration: totalDuration + existingDuration,
        message: 'Voice samples uploaded successfully'
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Start voice model training
 */
export const trainVoiceModel = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user.id;
    const { sampleIds } = req.body;
    
    if (!sampleIds || !Array.isArray(sampleIds) || sampleIds.length === 0) {
      throw new ApiError(400, 'Sample IDs are required');
    }
    
    // Start training
    const jobId = await synthesizer.startTraining(userId, sampleIds);
    
    res.status(200).json({
      success: true,
      data: {
        jobId,
        status: 'training_started',
        message: 'Voice model training has been initiated'
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Synthesize speech from text
 */
export const synthesizeSpeech = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user.id;
    const { text, emotion = 'neutral' } = req.body;
    
    if (!text) {
      throw new ApiError(400, 'Text is required');
    }
    
    if (text.length > 1000) {
      throw new ApiError(400, 'Text exceeds maximum length of 1000 characters');
    }
    
    // Generate speech
    const audioUrl = await synthesizer.synthesize(userId, text, emotion);
    
    res.status(200).json({
      success: true,
      data: {
        audioUrl,
        text,
        emotion
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Get voice profile status
 */
export const getVoiceProfile = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user.id;
    
    const profile = await synthesizer.getVoiceProfile(userId);
    
    if (!profile) {
      return res.status(200).json({
        success: true,
        data: {
          exists: false,
          message: 'Voice profile not found'
        }
      });
    }
    
    res.status(200).json({
      success: true,
      data: {
        exists: true,
        profile
      }
    });
  } catch (error) {
    next(error);
  }
};