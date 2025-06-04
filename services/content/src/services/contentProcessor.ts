import { createWorker } from 'tesseract.js'
import { createCanvas, loadImage } from 'canvas'

export class ContentProcessor {
  /**
   * Process uploaded file to extract metadata
   * @param file Uploaded file
   * @returns Metadata about the file
   */
  async processFile(file: Express.Multer.File): Promise<any> {
    const metadata: any = {
      fileType: file.mimetype,
      fileSize: file.size,
      createdAt: new Date().toISOString()
    }

    // Process based on file type
    if (file.mimetype.startsWith('image/')) {
      const imageMetadata = await this.processImage(file)
      Object.assign(metadata, imageMetadata)
    } else if (file.mimetype.startsWith('audio/')) {
      const audioMetadata = this.processAudio(file)
      Object.assign(metadata, audioMetadata)
    } else if (file.mimetype.startsWith('video/')) {
      const videoMetadata = this.processVideo(file)
      Object.assign(metadata, videoMetadata)
    } else if (
      file.mimetype === 'application/pdf' ||
      file.mimetype === 'text/plain' ||
      file.mimetype.includes('document')
    ) {
      const docMetadata = this.processDocument(file)
      Object.assign(metadata, docMetadata)
    }

    return metadata
  }

  /**
   * Process image file
   * @param file Image file
   * @returns Image metadata
   */
  private async processImage(file: Express.Multer.File): Promise<any> {
    try {
      // Extract text from image using OCR
      const worker = await createWorker('eng')
      const { data } = await worker.recognize(file.buffer)
      await worker.terminate()

      // Load image to get dimensions
      const img = await loadImage(file.buffer)
      const canvas = createCanvas(img.width, img.height)

      return {
        width: img.width,
        height: img.height,
        hasText: data.text.trim().length > 0,
        extractedText: data.text.trim(),
        confidence: data.confidence
      }
    } catch (error) {
      console.error('Error processing image:', error)
      return {
        processingError: 'Failed to process image'
      }
    }
  }

  /**
   * Process audio file
   * @param file Audio file
   * @returns Audio metadata
   */
  private processAudio(file: Express.Multer.File): any {
    // Basic audio metadata
    // In a real implementation, you would use a library like ffprobe to extract duration, codec, etc.
    return {
      format: file.mimetype.split('/')[1],
      sizeInMB: Math.round((file.size / (1024 * 1024)) * 100) / 100
    }
  }

  /**
   * Process video file
   * @param file Video file
   * @returns Video metadata
   */
  private processVideo(file: Express.Multer.File): any {
    // Basic video metadata
    // In a real implementation, you would use a library like ffprobe to extract duration, resolution, etc.
    return {
      format: file.mimetype.split('/')[1],
      sizeInMB: Math.round((file.size / (1024 * 1024)) * 100) / 100
    }
  }

  /**
   * Process document file
   * @param file Document file
   * @returns Document metadata
   */
  private processDocument(file: Express.Multer.File): any {
    // Basic document metadata
    // In a real implementation, you would use libraries specific to the document type
    return {
      format: file.mimetype.split('/')[1],
      sizeInMB: Math.round((file.size / (1024 * 1024)) * 100) / 100
    }
  }
}