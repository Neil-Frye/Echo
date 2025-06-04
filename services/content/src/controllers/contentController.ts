import { Request, Response, NextFunction } from 'express'
import { S3Service } from '../services/s3Service'
import { ContentProcessor } from '../services/contentProcessor'
import { db } from '../db'
import { v4 as uuidv4 } from 'uuid'

const s3Service = new S3Service()
const contentProcessor = new ContentProcessor()

export const uploadContent = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { userId } = req.user
    const { title, description, tags } = req.body
    const file = req.file

    if (!file) {
      return res.status(400).json({
        success: false,
        error: 'No file uploaded'
      })
    }

    // Upload to S3
    const s3Url = await s3Service.uploadFile(userId, file)
    
    // Process content for AI
    const metadata = await contentProcessor.processFile(file)
    
    // Save to database
    const contentId = uuidv4()
    await db.query(
      `INSERT INTO content_items 
       (id, user_id, content_type, title, description, s3_url, 
        file_size_bytes, metadata, tags, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())`,
      [
        contentId,
        userId,
        file.mimetype,
        title,
        description,
        s3Url,
        file.size,
        JSON.stringify(metadata),
        tags || []
      ]
    )

    res.json({
      success: true,
      data: {
        id: contentId,
        s3Url,
        metadata
      }
    })
  } catch (error) {
    next(error)
  }
}

export const getUserContent = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { userId } = req.user
    const { type, page = 1, limit = 20 } = req.query
    
    const offset = (Number(page) - 1) * Number(limit)
    
    let query = `
      SELECT * FROM content_items 
      WHERE user_id = $1
    `
    const params: any[] = [userId]
    
    if (type) {
      query += ` AND content_type = $2`
      params.push(type)
    }
    
    query += ` ORDER BY created_at DESC LIMIT ${params.length + 1} OFFSET ${params.length + 2}`
    params.push(Number(limit), offset)
    
    const result = await db.query(query, params)
    
    res.json({
      success: true,
      data: {
        items: result.rows,
        page: Number(page),
        limit: Number(limit),
        total: result.rowCount
      }
    })
  } catch (error) {
    next(error)
  }
}