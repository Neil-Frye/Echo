import { Request, Response, NextFunction } from 'express'

interface ValidationOptions {
  body?: string[]
  query?: string[]
  params?: string[]
}

export const validate = (options: ValidationOptions) => {
  return (req: Request, res: Response, next: NextFunction) => {
    const errors: string[] = []

    // Validate body fields
    if (options.body) {
      for (const field of options.body) {
        if (!req.body[field]) {
          errors.push(`Missing required field in body: ${field}`)
        }
      }
    }

    // Validate query params
    if (options.query) {
      for (const field of options.query) {
        if (!req.query[field]) {
          errors.push(`Missing required query parameter: ${field}`)
        }
      }
    }

    // Validate URL params
    if (options.params) {
      for (const field of options.params) {
        if (!req.params[field]) {
          errors.push(`Missing required URL parameter: ${field}`)
        }
      }
    }

    // If validation errors exist, return 400
    if (errors.length > 0) {
      return res.status(400).json({
        success: false,
        errors
      })
    }

    next()
  }
}