# TypeScript Style Guide

## 📋 Table of Contents

- [Introduction](#introduction)
- [General Principles](#general-principles)
- [File Organization](#file-organization)
- [Naming Conventions](#naming-conventions)
- [Type Definitions](#type-definitions)
- [Functions](#functions)
- [Classes](#classes)
- [React/Next.js Guidelines](#reactnextjs-guidelines)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Performance](#performance)
- [Security](#security)

## 🎯 Introduction

This style guide defines the TypeScript coding standards for EthernalEcho. Following these guidelines ensures consistent, maintainable, and high-quality code across our platform.

## 🏗️ General Principles

1. **Readability First**: Code is read more than it's written
2. **Type Safety**: Leverage TypeScript's type system fully
3. **Immutability**: Prefer immutable data structures
4. **Functional Programming**: Favor pure functions over side effects
5. **Explicit over Implicit**: Be clear about intentions

## 📁 File Organization

### Directory Structure

```
src/
├── components/          # React components
│   ├── common/         # Shared components
│   ├── features/       # Feature-specific components
│   └── layouts/        # Layout components
├── hooks/              # Custom React hooks
├── services/           # API services and external integrations
├── utils/              # Utility functions
├── types/              # TypeScript type definitions
├── constants/          # Application constants
└── styles/             # Global styles and themes
```

### File Naming

- **Components**: PascalCase (e.g., `VoiceRecorder.tsx`)
- **Utilities**: camelCase (e.g., `audioProcessor.ts`)
- **Types**: PascalCase with `.types.ts` suffix (e.g., `User.types.ts`)
- **Tests**: Same name with `.test.ts` suffix (e.g., `VoiceRecorder.test.tsx`)
- **Styles**: Same name with `.module.css` suffix (e.g., `VoiceRecorder.module.css`)

### Imports Order

```typescript
// 1. External dependencies
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';

// 2. Internal dependencies - absolute imports
import { Button } from ''components/common'' (see below for file content);
import { useAuth } from ''hooks'' (see below for file content);
import { ApiService } from ''services'' (see below for file content);

// 3. Internal dependencies - relative imports
import { VoiceProcessor } from './VoiceProcessor';
import { AudioVisualizer } from './AudioVisualizer';

// 4. Types
import type { VoiceRecorderProps } from './VoiceRecorder.types';

// 5. Styles
import styles from './VoiceRecorder.module.css';

// 6. Constants
import { AUDIO_CONSTRAINTS, MAX_RECORDING_DURATION } from './constants';
```

## 📝 Naming Conventions

### Variables and Functions

```typescript
// ✅ Good
const userProfile = await fetchUserProfile(userId);
const isRecording = false;
const maxRetryAttempts = 3;

function calculateVoiceQuality(samples: VoiceSample[]): number {
  // ...
}

// ❌ Bad
const prof = await get_prof(id);
const recording = false; // Ambiguous boolean name
const MAX_RETRY = 3; // Should be camelCase

function calc(s: any[]): number {
  // ...
}
```

### Types and Interfaces

```typescript
// ✅ Good
interface UserProfile {
  id: string;
  email: string;
  createdAt: Date;
}

type VoiceQuality = 'low' | 'medium' | 'high';

enum SubscriptionTier {
  Lite = 'LITE',
  Standard = 'STANDARD',
  Premium = 'PREMIUM',
  Lifetime = 'LIFETIME'
}

// ❌ Bad
interface user_profile {
  ID: string;
  Email: string;
  created: Date;
}

type Quality = string; // Too generic
```

### Constants

```typescript
// ✅ Good
export const API_ENDPOINTS = {
  AUTH: '/api/v1/auth',
  USERS: '/api/v1/users',
  VOICE: '/api/v1/voice'
} as const;

export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const SUPPORTED_AUDIO_FORMATS = ['mp3', 'wav', 'webm'] as const;

// ❌ Bad
export const auth_endpoint = '/api/v1/auth';
export const MAXSIZE = 10485760; // Magic number
```

## 🔧 Type Definitions

### Use Type Inference

```typescript
// ✅ Good - Let TypeScript infer when obvious
const count = 0;
const users = ['Alice', 'Bob'];
const isActive = true;

// ✅ Good - Explicit when not obvious
const delay: number = parseInt(process.env.DELAY || '1000');
const callback: (error: Error | null) => void = handleError;

// ❌ Bad - Redundant type annotations
const count: number = 0;
const users: string[] = ['Alice', 'Bob'];
```

### Prefer Interfaces for Objects

```typescript
// ✅ Good - Interface for object shapes
interface VoiceRecorderConfig {
  sampleRate: number;
  channels: number;
  bitDepth: number;
}

// ✅ Good - Type for unions/intersections
type AudioFormat = 'mp3' | 'wav' | 'webm';
type RecorderState = 'idle' | 'recording' | 'processing';

// ❌ Bad - Type for object shapes
type VoiceRecorderConfig = {
  sampleRate: number;
  channels: number;
  bitDepth: number;
};
```

### Utility Types

```typescript
// ✅ Good - Use built-in utility types
interface User {
  id: string;
  email: string;
  password: string;
  createdAt: Date;
}

type PublicUser = Omit<User, 'password'>;
type UserUpdate = Partial<Omit<User, 'id' | 'createdAt'>>;
type ReadonlyUser = Readonly<User>;

// Custom utility types for domain-specific needs
type Nullable<T> = T | null;
type AsyncFunction<T = void> = () => Promise<T>;
```

### Discriminated Unions

```typescript
// ✅ Good - Use discriminated unions for state
type VoiceProcessingState = 
  | { status: 'idle' }
  | { status: 'recording'; duration: number }
  | { status: 'processing'; progress: number }
  | { status: 'completed'; result: VoiceModel }
  | { status: 'error'; error: Error };

function handleVoiceState(state: VoiceProcessingState) {
  switch (state.status) {
    case 'recording':
      return `Recording: ${state.duration}s`;
    case 'processing':
      return `Processing: ${state.progress}%`;
    case 'completed':
      return `Completed: ${state.result.id}`;
    case 'error':
      return `Error: ${state.error.message}`;
    default:
      return 'Ready to record';
  }
}
```

## 🎯 Functions

### Function Declarations

```typescript
// ✅ Good - Named exports for better refactoring
export function processVoiceSample(
  audioData: Float32Array,
  sampleRate: number = 16000
): VoiceFeatures {
  // Implementation
}

// ✅ Good - Arrow functions for callbacks
const handleRecordingComplete = async (blob: Blob): Promise<void> => {
  await uploadVoiceSample(blob);
};

// ❌ Bad - Default exports make refactoring harder
export default function processVoiceSample() {
  // ...
}
```

### Function Parameters

```typescript
// ✅ Good - Use object parameters for multiple options
interface UploadOptions {
  file: File;
  onProgress?: (progress: number) => void;
  metadata?: Record<string, string>;
  timeout?: number;
}

async function uploadFile(options: UploadOptions): Promise<string> {
  const { file, onProgress, metadata = {}, timeout = 30000 } = options;
  // Implementation
}

// ❌ Bad - Too many positional parameters
async function uploadFile(
  file: File,
  onProgress?: (progress: number) => void,
  metadata?: Record<string, string>,
  timeout?: number
): Promise<string> {
  // Hard to use and remember parameter order
}
```

### Pure Functions

```typescript
// ✅ Good - Pure function
function calculateAudioDuration(
  sampleCount: number,
  sampleRate: number
): number {
  return sampleCount / sampleRate;
}

// ❌ Bad - Side effects
let totalDuration = 0;
function calculateAudioDuration(
  sampleCount: number,
  sampleRate: number
): number {
  const duration = sampleCount / sampleRate;
  totalDuration += duration; // Side effect!
  console.log(`Duration: ${duration}`); // Side effect!
  return duration;
}
```

## 🏛️ Classes

### Class Structure

```typescript
// ✅ Good - Well-structured class
export class VoiceRecorder {
  private readonly audioContext: AudioContext;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  
  constructor(
    private readonly config: VoiceRecorderConfig = DEFAULT_CONFIG
  ) {
    this.audioContext = new AudioContext();
  }
  
  async startRecording(): Promise<void> {
    // Method implementation
  }
  
  async stopRecording(): Promise<Blob> {
    // Method implementation
  }
  
  private processAudioChunks(): void {
    // Private method
  }
}

// ❌ Bad - Poor class structure
export class VoiceRecorder {
  audioContext: any; // No type
  config: any; // No type
  
  constructor(config: any) {
    this.config = config;
    this.init(); // Side effect in constructor
  }
  
  init() {
    // Should be in constructor
  }
  
  // Public method that should be private
  _processChunks() {
    // Naming convention violation
  }
}
```

### Method Chaining

```typescript
// ✅ Good - Fluent interface
export class VoiceModelBuilder {
  private config: Partial<VoiceModelConfig> = {};
  
  withSampleRate(rate: number): this {
    this.config.sampleRate = rate;
    return this;
  }
  
  withLanguage(language: string): this {
    this.config.language = language;
    return this;
  }
  
  withEmotionDetection(enabled: boolean = true): this {
    this.config.emotionDetection = enabled;
    return this;
  }
  
  build(): VoiceModel {
    return new VoiceModel(this.config as VoiceModelConfig);
  }
}

// Usage
const model = new VoiceModelBuilder()
  .withSampleRate(16000)
  .withLanguage('en-US')
  .withEmotionDetection()
  .build();
```

## ⚛️ React/Next.js Guidelines

### Component Structure

```typescript
// ✅ Good - Well-structured component
import React, { useState, useCallback, useEffect } from 'react';
import { useAuth } from ''hooks'' (see below for file content);
import type { VoiceRecorderProps } from './VoiceRecorder.types';

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onComplete,
  maxDuration = 300,
  className
}) => {
  // 1. Hooks
  const { user } = useAuth();
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  
  // 2. Effects
  useEffect(() => {
    // Effect implementation
  }, [dependency]);
  
  // 3. Callbacks
  const handleStart = useCallback(() => {
    setIsRecording(true);
    // Start logic
  }, []);
  
  const handleStop = useCallback(() => {
    setIsRecording(false);
    // Stop logic
  }, []);
  
  // 4. Render
  return (
    <div className={className}>
      {/* Component JSX */}
    </div>
  );
};

// ❌ Bad - Poor component structure
function VoiceRecorder(props: any) {
  const [recording, setRecording] = useState();
  
  // Logic mixed with render
  if (recording) {
    // Logic that should be in useEffect
  }
  
  function start() {
    // Not memoized
  }
  
  return <div>{/* ... */}</div>;
}
```

### Custom Hooks

```typescript
// ✅ Good - Well-designed custom hook
export function useVoiceRecorder(options?: UseVoiceRecorderOptions) {
  const [state, setState] = useState<VoiceRecorderState>({
    status: 'idle',
    duration: 0,
    audioBlob: null,
    error: null
  });
  
  const startRecording = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, status: 'recording', error: null }));
      // Implementation
    } catch (error) {
      setState(prev => ({ ...prev, status: 'error', error }));
    }
  }, []);
  
  const stopRecording = useCallback(async () => {
    // Implementation
  }, []);
  
  return {
    ...state,
    startRecording,
    stopRecording,
    isRecording: state.status === 'recording'
  };
}

// Usage
function MyComponent() {
  const { isRecording, startRecording, stopRecording } = useVoiceRecorder();
  // Use the hook
}
```

### Props and State

```typescript
// ✅ Good - Properly typed props
interface UserProfileProps {
  userId: string;
  onUpdate?: (profile: UserProfile) => void;
  className?: string;
  children?: React.ReactNode;
}

// ✅ Good - Discriminated union for complex state
type FormState = 
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: UserProfile }
  | { status: 'error'; error: Error };

// ✅ Good - State reducer
type Action = 
  | { type: 'FETCH_START' }
  | { type: 'FETCH_SUCCESS'; payload: UserProfile }
  | { type: 'FETCH_ERROR'; payload: Error };

function profileReducer(state: FormState, action: Action): FormState {
  switch (action.type) {
    case 'FETCH_START':
      return { status: 'loading' };
    case 'FETCH_SUCCESS':
      return { status: 'success', data: action.payload };
    case 'FETCH_ERROR':
      return { status: 'error', error: action.payload };
    default:
      return state;
  }
}
```

## 🚨 Error Handling

### Error Types

```typescript
// ✅ Good - Custom error classes
export class VoiceRecordingError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly details?: unknown
  ) {
    super(message);
    this.name = 'VoiceRecordingError';
  }
}

export class AuthenticationError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number = 401
  ) {
    super(message);
    this.name = 'AuthenticationError';
  }
}

// Usage
throw new VoiceRecordingError(
  'Microphone access denied',
  'MIC_PERMISSION_DENIED',
  { browser: navigator.userAgent }
);
```

### Try-Catch Patterns

```typescript
// ✅ Good - Proper error handling
export async function uploadVoiceSample(
  file: File
): Promise<UploadResult> {
  try {
    validateFile(file);
    
    const formData = new FormData();
    formData.append('audio', file);
    
    const response = await fetch('/api/voice/upload', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new ApiError(
        `Upload failed: ${response.statusText}`,
        response.status
      );
    }
    
    return await response.json();
  } catch (error) {
    // Log error for monitoring
    logger.error('Voice upload failed', { error, file: file.name });
    
    // Re-throw with context
    if (error instanceof ApiError) {
      throw error;
    }
    
    throw new VoiceRecordingError(
      'Failed to upload voice sample',
      'UPLOAD_FAILED',
      error
    );
  }
}

// ❌ Bad - Poor error handling
async function uploadVoiceSample(file: File) {
  try {
    const response = await fetch('/api/voice/upload', {
      method: 'POST',
      body: file // Wrong format
    });
    return response.json(); // No error checking
  } catch (e) {
    console.log(e); // Just logging
    return null; // Hiding the error
  }
}
```

### Result Types

```typescript
// ✅ Good - Result type for explicit error handling
type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

async function parseVoiceFeatures(
  audioData: ArrayBuffer
): Promise<Result<VoiceFeatures>> {
  try {
    const features = await extractFeatures(audioData);
    return { success: true, data: features };
  } catch (error) {
    return { success: false, error: error as Error };
  }
}

// Usage
const result = await parseVoiceFeatures(audioData);
if (result.success) {
  console.log('Features:', result.data);
} else {
  console.error('Failed:', result.error.message);
}
```

## 🧪 Testing

### Test Structure

```typescript
// ✅ Good - Well-structured tests
import { renderHook, act } from '@testing-library/react-hooks';
import { useVoiceRecorder } from './useVoiceRecorder';

describe('useVoiceRecorder', () => {
  let mockMediaRecorder: jest.Mocked<MediaRecorder>;
  
  beforeEach(() => {
    // Setup mocks
    mockMediaRecorder = createMockMediaRecorder();
    global.MediaRecorder = jest.fn(() => mockMediaRecorder);
  });
  
  afterEach(() => {
    jest.clearAllMocks();
  });
  
  describe('startRecording', () => {
    it('should start recording when called', async () => {
      const { result } = renderHook(() => useVoiceRecorder());
      
      await act(async () => {
        await result.current.startRecording();
      });
      
      expect(result.current.isRecording).toBe(true);
      expect(mockMediaRecorder.start).toHaveBeenCalledTimes(1);
    });
    
    it('should handle permission denial gracefully', async () => {
      // Mock permission denial
      jest.spyOn(navigator.mediaDevices, 'getUserMedia')
        .mockRejectedValueOnce(new Error('Permission denied'));
      
      const { result } = renderHook(() => useVoiceRecorder());
      
      await act(async () => {
        await result.current.startRecording();
      });
      
      expect(result.current.error).toEqual(
        expect.objectContaining({
          message: expect.stringContaining('Permission denied')
        })
      );
    });
  });
});
```

### Test Utilities

```typescript
// ✅ Good - Reusable test utilities
export const createMockUser = (overrides?: Partial<User>): User => ({
  id: 'test-user-id',
  email: 'test@example.com',
  createdAt: new Date('2024-01-01'),
  ...overrides
});

export const createMockVoiceSample = (
  duration: number = 30
): VoiceSample => ({
  id: 'test-sample-id',
  duration,
  quality: 0.95,
  audioUrl: 'https://example.com/audio.mp3'
});

export const waitForAsync = (ms: number = 0): Promise<void> => 
  new Promise(resolve => setTimeout(resolve, ms));
```

## ⚡ Performance

### Memoization

```typescript
// ✅ Good - Proper memoization
import { useMemo, useCallback, memo } from 'react';

export const VoiceWaveform = memo<VoiceWaveformProps>(({
  audioData,
  width,
  height,
  color = '#3B82F6'
}) => {
  // Expensive calculation memoized
  const peaks = useMemo(() => 
    calculateWaveformPeaks(audioData, width),
    [audioData, width]
  );
  
  // Callback memoized
  const handleClick = useCallback((index: number) => {
    const time = (index / width) * duration;
    seekTo(time);
  }, [width, duration]);
  
  return (
    <canvas
      width={width}
      height={height}
      onClick={handleClick}
    />
  );
});

VoiceWaveform.displayName = 'VoiceWaveform';
```

### Lazy Loading

```typescript
// ✅ Good - Code splitting
import { lazy, Suspense } from 'react';

const VoiceAnalyzer = lazy(() => 
  import('./VoiceAnalyzer')
    .then(module => ({ default: module.VoiceAnalyzer }))
);

export function AnalyzerSection() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <VoiceAnalyzer />
    </Suspense>
  );
}

// ✅ Good - Dynamic imports
async function loadAudioProcessor() {
  const { AudioProcessor } = await import(''utils/audio')' (see below for file content);
  return new AudioProcessor();
}
```

## 🔒 Security

### Input Validation

```typescript
// ✅ Good - Proper validation
import { z } from 'zod';

const VoiceUploadSchema = z.object({
  file: z.instanceof(File).refine(
    file => file.size <= 10 * 1024 * 1024,
    'File must be less than 10MB'
  ).refine(
    file => ['audio/mp3', 'audio/wav', 'audio/webm'].includes(file.type),
    'Invalid file type'
  ),
  metadata: z.object({
    duration: z.number().positive().max(300),
    language: z.string().length(2),
    quality: z.enum(['low', 'medium', 'high'])
  })
});

export function validateVoiceUpload(data: unknown) {
  return VoiceUploadSchema.parse(data);
}
```

### Secure Data Handling

```typescript
// ✅ Good - Secure practices
export class SecureStorage {
  private readonly encryption: EncryptionService;
  
  async store(key: string, data: unknown): Promise<void> {
    const serialized = JSON.stringify(data);
    const encrypted = await this.encryption.encrypt(serialized);
    localStorage.setItem(key, encrypted);
  }
  
  async retrieve<T>(key: string): Promise<T | null> {
    const encrypted = localStorage.getItem(key);
    if (!encrypted) return null;
    
    try {
      const decrypted = await this.encryption.decrypt(encrypted);
      return JSON.parse(decrypted) as T;
    } catch (error) {
      // Log but don't expose error details
      console.error('Failed to retrieve secure data');
      return null;
    }
  }
}

// ❌ Bad - Insecure practices
function storeUserData(data: any) {
  // No validation
  localStorage.setItem('user', JSON.stringify(data));
  
  // Logging sensitive data
  console.log('Stored user data:', data);
}
```

## 📚 Resources

- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)
- [Effective TypeScript](https://effectivetypescript.com/)

---

*Last updated: January 2024*
