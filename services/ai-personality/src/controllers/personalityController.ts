import { Request, Response, NextFunction } from 'express'
import { spawn } from 'child_process'
import path from 'path'
import { db } from '../db'

export const personalityController = {
  /**
   * Train a personality model for a user
   */
  trainPersonalityModel: async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      const { userId } = req.user
      const { contentIds } = req.body

      // Check if there are enough content items
      if (!contentIds || contentIds.length < 5) {
        return res.status(400).json({
          success: false,
          error: 'At least 5 content items are required for personality training'
        })
      }

      // Fetch content items
      const contentQuery = `
        SELECT * FROM content_items 
        WHERE id = ANY($1) AND user_id = $2
      `
      const contentResult = await db.query(contentQuery, [contentIds, userId])

      if (contentResult.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'No content items found'
        })
      }

      // Fetch conversations
      const conversationsQuery = `
        SELECT * FROM conversations 
        WHERE user_id = $1 
        ORDER BY created_at DESC 
        LIMIT 50
      `
      const conversationsResult = await db.query(conversationsQuery, [userId])

      // Prepare training data
      const trainingData = {
        texts: contentResult.rows.map(item => item.metadata?.extractedText || ''),
        conversations: conversationsResult.rows.map(conv => ({
          id: conv.id,
          messages: conv.messages || []
        })),
        memories: contentResult.rows.map(item => ({
          id: item.id,
          text: item.metadata?.extractedText || item.description || '',
          source: 'content',
          contentType: item.content_type,
          createdAt: item.created_at
        }))
      }

      // Launch Python process for model training
      const pythonProcess = spawn('python', [
        path.join(__dirname, '../python/personality_model.py'),
        JSON.stringify({
          action: 'train',
          userId,
          trainingData
        })
      ])

      let result = ''
      pythonProcess.stdout.on('data', (data) => {
        result += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`)
      })

      pythonProcess.on('close', async (code) => {
        if (code !== 0) {
          return res.status(500).json({
            success: false,
            error: 'Model training failed'
          })
        }

        try {
          const modelResult = JSON.parse(result)
          
          if (!modelResult.success) {
            return res.status(500).json({
              success: false,
              error: modelResult.error || 'Model training failed'
            })
          }

          // Update user's personality model status
          await db.query(
            `INSERT INTO personality_models 
             (user_id, model_path, status, traits, training_completed_at)
             VALUES ($1, $2, $3, $4, NOW())
             ON CONFLICT (user_id) 
             DO UPDATE SET 
               model_path = $2,
               status = $3,
               traits = $4,
               training_completed_at = NOW()`,
            [userId, modelResult.model_path, 'ready', JSON.stringify(modelResult.traits)]
          )

          return res.json({
            success: true,
            data: {
              traits: modelResult.traits,
              status: 'ready'
            }
          })
        } catch (error) {
          console.error('Error parsing Python result:', error)
          return res.status(500).json({
            success: false,
            error: 'Failed to parse model training result'
          })
        }
      })
    } catch (error) {
      next(error)
    }
  },

  /**
   * Generate a response based on the user's personality model
   */
  generateResponse: async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      const { userId } = req.user
      const { message, conversationId } = req.body

      // Check if user has a trained model
      const modelQuery = `
        SELECT * FROM personality_models 
        WHERE user_id = $1 AND status = 'ready'
      `
      const modelResult = await db.query(modelQuery, [userId])

      if (modelResult.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'No trained personality model found'
        })
      }

      // Get conversation history
      const conversationQuery = `
        SELECT * FROM conversations 
        WHERE id = $1 AND user_id = $2
      `
      const conversationResult = await db.query(conversationQuery, [conversationId, userId])

      if (conversationResult.rows.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'Conversation not found'
        })
      }

      const conversation = conversationResult.rows[0]

      // Prepare context for response generation
      const context = {
        conversation_history: conversation.messages || [],
        memories: [] // Could load relevant memories here
      }

      // Launch Python process for response generation
      const pythonProcess = spawn('python', [
        path.join(__dirname, '../python/personality_model.py'),
        JSON.stringify({
          action: 'generate',
          userId,
          message,
          context
        })
      ])

      let result = ''
      pythonProcess.stdout.on('data', (data) => {
        result += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`)
      })

      pythonProcess.on('close', async (code) => {
        if (code !== 0) {
          return res.status(500).json({
            success: false,
            error: 'Response generation failed'
          })
        }

        try {
          const responseResult = JSON.parse(result)
          
          if (!responseResult.success) {
            return res.status(500).json({
              success: false,
              error: responseResult.error || 'Response generation failed'
            })
          }

          // Add message to conversation
          const newMessages = [
            ...conversation.messages,
            { sender: 'human', text: message, timestamp: new Date() },
            { sender: 'assistant', text: responseResult.response, timestamp: new Date() }
          ]

          // Update conversation in database
          await db.query(
            `UPDATE conversations 
             SET messages = $1, updated_at = NOW()
             WHERE id = $2`,
            [JSON.stringify(newMessages), conversationId]
          )

          return res.json({
            success: true,
            data: {
              response: responseResult.response,
              conversationId
            }
          })
        } catch (error) {
          console.error('Error parsing Python result:', error)
          return res.status(500).json({
            success: false,
            error: 'Failed to parse response generation result'
          })
        }
      })
    } catch (error) {
      next(error)
    }
  },

  /**
   * Get the user's personality profile
   */
  getPersonalityProfile: async (
    req: Request,
    res: Response,
    next: NextFunction
  ) => {
    try {
      const { userId } = req.user

      // Get personality model
      const modelQuery = `
        SELECT * FROM personality_models 
        WHERE user_id = $1
      `
      const modelResult = await db.query(modelQuery, [userId])

      if (modelResult.rows.length === 0) {
        return res.json({
          success: true,
          data: {
            hasModel: false,
            traits: null,
            status: 'not_trained'
          }
        })
      }

      const model = modelResult.rows[0]

      return res.json({
        success: true,
        data: {
          hasModel: true,
          traits: model.traits,
          status: model.status,
          lastUpdated: model.training_completed_at
        }
      })
    } catch (error) {
      next(error)
    }
  }
}