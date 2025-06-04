import { S3 } from 'aws-sdk'
import { v4 as uuidv4 } from 'uuid'

export class S3Service {
  private s3: S3

  constructor() {
    this.s3 = new S3({
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
      region: process.env.AWS_REGION || 'us-east-1'
    })
  }

  /**
   * Upload a file to S3
   * @param userId User ID
   * @param file File to upload
   * @returns S3 URL of the uploaded file
   */
  async uploadFile(userId: string, file: Express.Multer.File): Promise<string> {
    const fileId = uuidv4()
    const key = `users/${userId}/content/${fileId}-${file.originalname}`
    
    const params = {
      Bucket: process.env.AWS_S3_BUCKET || 'echo-content',
      Key: key,
      Body: file.buffer,
      ContentType: file.mimetype,
      ACL: 'private'
    }

    const result = await this.s3.upload(params).promise()
    
    return result.Location
  }

  /**
   * Generate a signed URL for accessing a private file
   * @param s3Url S3 URL of the file
   * @param expiresInSeconds Time in seconds until the URL expires
   * @returns Signed URL
   */
  async getSignedUrl(s3Url: string, expiresInSeconds = 3600): Promise<string> {
    // Extract bucket and key from the S3 URL
    const url = new URL(s3Url)
    const bucket = url.hostname.split('.')[0]
    const key = url.pathname.substring(1) // Remove leading slash
    
    const params = {
      Bucket: bucket,
      Key: key,
      Expires: expiresInSeconds
    }
    
    return this.s3.getSignedUrlPromise('getObject', params)
  }
}