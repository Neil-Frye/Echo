// API Constants
export const API_VERSION = 'v1';
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_PAGE_SIZE = 100;

// Content Constants
export const MAX_UPLOAD_SIZE_MB = 500;
export const ALLOWED_AUDIO_FORMATS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a'];
export const ALLOWED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
export const ALLOWED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
export const ALLOWED_DOCUMENT_FORMATS = ['.pdf', '.doc', '.docx', '.txt', '.rtf'];

// Voice Training Constants
export const MIN_VOICE_SAMPLES = 5;
export const RECOMMENDED_VOICE_SAMPLES = 20;
export const MIN_TOTAL_AUDIO_SECONDS = 60;
export const RECOMMENDED_TOTAL_AUDIO_SECONDS = 300;

// Personality Constants
export const PERSONALITY_TRAITS = [
  'openness',
  'conscientiousness',
  'extraversion',
  'agreeableness',
  'neuroticism',
  'humor',
  'empathy',
  'creativity',
  'patience',
  'assertiveness',
  'adventurousness',
  'intellect',
  'emotional_stability',
  'warmth'
];

// Access Level Permissions
export const ACCESS_LEVEL_PERMISSIONS = {
  viewer: {
    canChat: true,
    canDownload: false,
    canEdit: false,
    canInvite: false,
    canDelete: false,
  },
  contributor: {
    canChat: true,
    canDownload: true,
    canEdit: false,
    canInvite: false,
    canDelete: false,
  },
  manager: {
    canChat: true,
    canDownload: true,
    canEdit: true,
    canInvite: true,
    canDelete: false,
  },
  owner: {
    canChat: true,
    canDownload: true,
    canEdit: true,
    canInvite: true,
    canDelete: true,
  },
};

// Subscription Plan Limits
export const SUBSCRIPTION_LIMITS = {
  free: {
    storageLimitGb: 5,
    familyMemberLimit: 2,
    audioDurationLimit: 60, // minutes
    conversationsPerMonth: 100,
  },
  personal: {
    basic: {
      storageLimitGb: 25,
      familyMemberLimit: 5,
      audioDurationLimit: 300, // minutes
      conversationsPerMonth: 500,
    },
    standard: {
      storageLimitGb: 100,
      familyMemberLimit: 10,
      audioDurationLimit: 1000, // minutes
      conversationsPerMonth: 2000,
    },
    premium: {
      storageLimitGb: 500,
      familyMemberLimit: 20,
      audioDurationLimit: -1, // unlimited
      conversationsPerMonth: -1, // unlimited
    },
  },
  family: {
    basic: {
      storageLimitGb: 50,
      familyMemberLimit: 10,
      audioDurationLimit: 600, // minutes
      conversationsPerMonth: 1000,
    },
    standard: {
      storageLimitGb: 200,
      familyMemberLimit: 20,
      audioDurationLimit: 2000, // minutes
      conversationsPerMonth: 5000,
    },
    premium: {
      storageLimitGb: 1000,
      familyMemberLimit: 50,
      audioDurationLimit: -1, // unlimited
      conversationsPerMonth: -1, // unlimited
    },
  },
  legacy: {
    standard: {
      storageLimitGb: 1000,
      familyMemberLimit: 100,
      audioDurationLimit: -1, // unlimited
      conversationsPerMonth: -1, // unlimited
    },
    premium: {
      storageLimitGb: 5000,
      familyMemberLimit: 250,
      audioDurationLimit: -1, // unlimited
      conversationsPerMonth: -1, // unlimited
    },
  },
};

// Error Codes
export const ERROR_CODES = {
  // Authentication Errors
  AUTH_INVALID_CREDENTIALS: 'auth/invalid-credentials',
  AUTH_EMAIL_NOT_VERIFIED: 'auth/email-not-verified',
  AUTH_ACCOUNT_DISABLED: 'auth/account-disabled',
  AUTH_EXPIRED_TOKEN: 'auth/expired-token',
  AUTH_INVALID_TOKEN: 'auth/invalid-token',
  AUTH_MISSING_TOKEN: 'auth/missing-token',
  AUTH_INSUFFICIENT_PERMISSIONS: 'auth/insufficient-permissions',
  
  // User Errors
  USER_NOT_FOUND: 'user/not-found',
  USER_ALREADY_EXISTS: 'user/already-exists',
  USER_INVALID_INPUT: 'user/invalid-input',
  
  // Subscription Errors
  SUBSCRIPTION_EXPIRED: 'subscription/expired',
  SUBSCRIPTION_LIMIT_EXCEEDED: 'subscription/limit-exceeded',
  SUBSCRIPTION_PAYMENT_FAILED: 'subscription/payment-failed',
  
  // Content Errors
  CONTENT_NOT_FOUND: 'content/not-found',
  CONTENT_INVALID_TYPE: 'content/invalid-type',
  CONTENT_TOO_LARGE: 'content/too-large',
  CONTENT_UPLOAD_FAILED: 'content/upload-failed',
  
  // AI Model Errors
  AI_MODEL_NOT_FOUND: 'ai-model/not-found',
  AI_MODEL_TRAINING_FAILED: 'ai-model/training-failed',
  AI_MODEL_INSUFFICIENT_DATA: 'ai-model/insufficient-data',
  
  // Family Member Errors
  FAMILY_MEMBER_NOT_FOUND: 'family-member/not-found',
  FAMILY_MEMBER_LIMIT_EXCEEDED: 'family-member/limit-exceeded',
  FAMILY_MEMBER_INVALID_EMAIL: 'family-member/invalid-email',
  
  // General Errors
  INTERNAL_SERVER_ERROR: 'general/internal-server-error',
  INVALID_REQUEST: 'general/invalid-request',
  RESOURCE_NOT_FOUND: 'general/resource-not-found',
  SERVICE_UNAVAILABLE: 'general/service-unavailable',
};