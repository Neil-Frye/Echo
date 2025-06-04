import { Request, Response, NextFunction } from 'express';
import { ApiResponse, ERROR_CODES } from '@ethernalecho/shared';

/**
 * Global error handler middleware
 */
export const errorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  console.error(`[Error] ${error.message}`, error);
  
  // Default error response
  const response: ApiResponse<null> = {
    success: false,
    error: {
      code: ERROR_CODES.INTERNAL_SERVER_ERROR,
      message: 'An unexpected error occurred'
    }
  };
  
  // Return error response
  res.status(500).json(response);
};