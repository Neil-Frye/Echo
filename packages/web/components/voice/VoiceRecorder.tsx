'use client'

import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { uploadVoiceSamples } from '@/lib/api/voice'

interface VoiceRecorderProps {
  userId: string
  onComplete: () => void
}

export function VoiceRecorder({ userId, onComplete }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordings, setRecordings] = useState<Blob[]>([])
  const [totalDuration, setTotalDuration] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        } 
      })

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setRecordings(prev => [...prev, blob])
        setTotalDuration(prev => prev + 30) // Add 30 seconds per recording
        
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)

      // Auto-stop after 30 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          mediaRecorderRef.current.stop()
          setIsRecording(false)
        }
      }, 30000)
    } catch (error) {
      console.error('Error starting recording:', error)
      alert('Please grant microphone access to record your voice.')
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [])

  const uploadRecordings = async () => {
    const formData = new FormData()
    recordings.forEach((blob, index) => {
      formData.append('samples', blob, `recording-${index}.webm`)
    })

    try {
      await uploadVoiceSamples(formData)
      onComplete()
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload voice samples. Please try again.')
    }
  }

  const progress = Math.min((totalDuration / 1800) * 100, 100)

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Voice Training</h2>
      <p className="text-gray-600 mb-6">
        Record at least 30 minutes of your voice to create your digital legacy.
      </p>

      <div className="mb-6">
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium">Progress</span>
          <span className="text-sm text-gray-500">
            {Math.floor(totalDuration / 60)} / 30 minutes
          </span>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="flex justify-center mb-6">
        {isRecording ? (
          <Button
            onClick={stopRecording}
            size="lg"
            variant="destructive"
            className="rounded-full w-24 h-24"
          >
            Stop
          </Button>
        ) : (
          <Button
            onClick={startRecording}
            size="lg"
            className="rounded-full w-24 h-24"
          >
            Record
          </Button>
        )}
      </div>

      <div className="space-y-4">
        <h3 className="font-semibold">Recording Tips:</h3>
        <ul className="list-disc list-inside space-y-2 text-sm text-gray-600">
          <li>Find a quiet environment without background noise</li>
          <li>Speak naturally as you would in conversation</li>
          <li>Include different emotions and tones</li>
          <li>Read the provided scripts and share personal stories</li>
        </ul>
      </div>

      {recordings.length > 0 && (
        <div className="mt-6 p-4 bg-gray-50 rounded">
          <p className="text-sm font-medium mb-2">
            Recorded Samples: {recordings.length}
          </p>
          {totalDuration >= 1800 && (
            <Button onClick={uploadRecordings} className="w-full">
              Upload Voice Samples
            </Button>
          )}
        </div>
      )}
    </div>
  )
}