import { API_URL } from '../constants'

/**
 * Upload voice samples to the server
 * @param samples FormData containing voice recordings
 * @returns Promise with the response
 */
export async function uploadVoiceSamples(samples: FormData) {
  const response = await fetch(`${API_URL}/api/v1/voice/upload`, {
    method: 'POST',
    credentials: 'include',
    body: samples
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to upload voice samples')
  }

  return response.json()
}

/**
 * Get voice training status
 * @returns Promise with the voice training status
 */
export async function getVoiceTrainingStatus() {
  const response = await fetch(`${API_URL}/api/v1/voice/profile`, {
    credentials: 'include'
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to get voice training status')
  }

  return response.json()
}

/**
 * Start voice model training
 * @param sampleIds IDs of the uploaded voice samples to use for training
 * @returns Promise with the training job information
 */
export async function startVoiceTraining(sampleIds: string[]) {
  const response = await fetch(`${API_URL}/api/v1/voice/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    credentials: 'include',
    body: JSON.stringify({ sampleIds })
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to start voice training')
  }

  return response.json()
}

/**
 * Synthesize speech from text using the user's voice model
 * @param text Text to convert to speech
 * @param emotion Optional emotion parameter (default: 'neutral')
 * @returns Promise with the synthesized audio URL
 */
export async function synthesizeSpeech(text: string, emotion: string = 'neutral') {
  const response = await fetch(`${API_URL}/api/v1/voice/synthesize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    credentials: 'include',
    body: JSON.stringify({ text, emotion })
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.message || 'Failed to synthesize speech')
  }

  return response.json()
}