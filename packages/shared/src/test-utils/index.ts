import { User, VoiceProfile, Subscription } from '../types'

export const createMockUser = (overrides?: Partial<User>): User => ({
  id: 'test-user-123',
  email: 'test@example.com',
  createdAt: new Date(),
  updatedAt: new Date(),
  emailVerified: true,
  twoFactorEnabled: false,
  status: 'active',
  ...overrides
})

export const createMockVoiceProfile = (overrides?: Partial<VoiceProfile>): VoiceProfile => ({
  id: 'voice-profile-123',
  userId: 'test-user-123',
  sampleCount: 30,
  totalDurationSeconds: 1800,
  qualityScore: 0.85,
  isVerified: true,
  trainingStatus: 'completed',
  modelVersion: '1.0',
  voiceModelUrl: 'https://bucket.s3.amazonaws.com/models/voice-123.bin',
  createdAt: new Date(),
  updatedAt: new Date(),
  ...overrides
})

export const createMockSubscription = (overrides?: Partial<Subscription>): Subscription => ({
  id: 'sub-123',
  userId: 'test-user-123',
  planType: 'monthly',
  tier: 'premium',
  status: 'active',
  currentPeriodStart: new Date(),
  currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
  cancelAtPeriodEnd: false,
  storageLimit: 100,
  familyMemberLimit: 10,
  createdAt: new Date(),
  updatedAt: new Date(),
  ...overrides
})