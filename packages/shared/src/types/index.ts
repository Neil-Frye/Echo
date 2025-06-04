// User and Authentication Types
export interface User {
  id: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
  emailVerified: boolean;
  twoFactorEnabled: boolean;
  status: UserStatus;
  deathCertificateUrl?: string;
  deathVerifiedAt?: Date;
}

export enum UserStatus {
  Active = 'active',
  Inactive = 'inactive',
  Suspended = 'suspended',
  Deceased = 'deceased'
}

// Subscription Types
export interface Subscription {
  id: string;
  userId: string;
  planType: SubscriptionPlanType;
  tier: SubscriptionTier;
  status: SubscriptionStatus;
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
  storageLimitGb: number;
  familyMemberLimit: number;
  createdAt: Date;
  updatedAt: Date;
}

export enum SubscriptionPlanType {
  Free = 'free',
  Personal = 'personal',
  Family = 'family',
  Legacy = 'legacy'
}

export enum SubscriptionTier {
  Basic = 'basic',
  Standard = 'standard',
  Premium = 'premium'
}

export enum SubscriptionStatus {
  Active = 'active',
  Canceled = 'canceled',
  PastDue = 'past_due',
  Unpaid = 'unpaid',
  Trialing = 'trialing'
}

// Profile Types
export interface Profile {
  id: string;
  userId: string;
  fullName: string;
  dateOfBirth?: Date;
  dateOfDeath?: Date;
  biography?: string;
  personalityTraits?: Record<string, any>;
  interests?: string[];
  values?: string[];
  speechPatterns?: Record<string, any>;
  medicalCondition?: string;
  urgencyLevel?: UrgencyLevel;
  createdAt: Date;
  updatedAt: Date;
}

export enum UrgencyLevel {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Critical = 'critical'
}

// Voice Profile Types
export interface VoiceProfile {
  id: string;
  userId: string;
  voiceModelUrl?: string;
  sampleCount: number;
  totalDurationSeconds: number;
  qualityScore?: number;
  isVerified: boolean;
  trainingStatus: TrainingStatus;
  modelVersion?: string;
  createdAt: Date;
  updatedAt: Date;
}

export enum TrainingStatus {
  NotStarted = 'not_started',
  InProgress = 'in_progress',
  Completed = 'completed',
  Failed = 'failed'
}

// AI Model Types
export interface AIModel {
  id: string;
  userId: string;
  modelType: AIModelType;
  modelUrl?: string;
  trainingDataUrl?: string;
  version: number;
  accuracyScore?: number;
  interactionCount: number;
  lastTrainedAt?: Date;
  isActive: boolean;
  createdAt: Date;
}

export enum AIModelType {
  Voice = 'voice',
  Personality = 'personality',
  Combined = 'combined'
}

// Content Item Types
export interface ContentItem {
  id: string;
  userId: string;
  contentType: ContentType;
  title?: string;
  description?: string;
  s3Url: string;
  fileSizeBytes: number;
  durationSeconds?: number;
  metadata?: Record<string, any>;
  isProcessed: boolean;
  aiAnalysis?: Record<string, any>;
  createdAt: Date;
  tags?: string[];
}

export enum ContentType {
  Audio = 'audio',
  Video = 'video',
  Image = 'image',
  Document = 'document',
  Memory = 'memory',
  Journal = 'journal'
}

// Family Member Types
export interface FamilyMember {
  id: string;
  userId: string;
  email: string;
  fullName?: string;
  relationship?: string;
  accessLevel: AccessLevel;
  invitationStatus: InvitationStatus;
  invitationSentAt?: Date;
  acceptedAt?: Date;
  canDownloadContent: boolean;
  canModifyAI: boolean;
  activationDate?: Date;
  createdAt: Date;
}

export enum AccessLevel {
  Viewer = 'viewer',
  Contributor = 'contributor',
  Manager = 'manager',
  Owner = 'owner'
}

export enum InvitationStatus {
  Pending = 'pending',
  Accepted = 'accepted',
  Declined = 'declined',
  Revoked = 'revoked',
  Expired = 'expired'
}

// Conversation Types
export interface Conversation {
  id: string;
  aiModelId: string;
  familyMemberId: string;
  startedAt: Date;
  endedAt?: Date;
  durationSeconds?: number;
  messageCount: number;
  sentimentScore?: number;
  satisfactionRating?: number;
  messages?: ConversationMessage[];
}

export interface ConversationMessage {
  id: string;
  conversationId: string;
  senderId: string;
  senderType: SenderType;
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export enum SenderType {
  Human = 'human',
  AI = 'ai'
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

// Pagination Types
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
  sort?: string;
  order?: 'asc' | 'desc';
}