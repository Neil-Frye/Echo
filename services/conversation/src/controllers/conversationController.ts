import { Request, Response, NextFunction } from 'express'

export const getConversation = (req: Request, res: Response, next: NextFunction) => {
  try {
    // Logic to get a conversation
    res.json({ message: `Get conversation ${req.params.id}` })
  } catch (error) {
    next(error)
  }
}

export const createConversation = (req: Request, res: Response, next: NextFunction) => {
  try {
    // Logic to create a conversation
    res.json({ message: 'Create conversation', data: req.body })
  } catch (error) {
    next(error)
  }
}
