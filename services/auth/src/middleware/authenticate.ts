import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { ApiResponse, ERROR_CODES } from '@ethernalecho/shared';

// JWT configuration
const JWT_SECRET = process.env.JWT_SECRET || 'default-jwt-secret';

// Extend Request type to include user
declare global {
  namespace Express {
    interface Request {
      user?: {
        userId: string;
        email: string;
      };
    }
  }
}

/**
 * Authentication middleware
 * Verifies JWT token from Authorization header
 */
export const authenticate = (req: Request, res: Response, next: NextFunction) => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;
    
    if (!authHeader) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_MISSING_TOKEN,
          message: 'Authentication token is required'
        }
      };
      return res.status(401).json(response);
    }
    
    // Check for Bearer token
    const parts = authHeader.split(' ');
    if (parts.length !== 2 || parts[0] !== 'Bearer') {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid authentication token format'
        }
      };
      return res.status(401).json(response);
    }
    
    const token = parts[1];
    
    // Verify token
    const decoded = jwt.verify(token, JWT_SECRET) as {
      userId: string;
      email: string;
    };
    
    // Attach user to request
    req.user = {
      userId: decoded.userId,
      email: decoded.email
    };
    
    next();
  } catch (error) {
    if (error instanceof jwt.JsonWebTokenError) {
      // Token verification failed
      if (error instanceof jwt.TokenExpiredError) {
        const response: ApiResponse<null> = {
          success: false,
          error: {
            code: ERROR_CODES.AUTH_EXPIRED_TOKEN,
            message: 'Authentication token has expired'
          }
        };
        return res.status(401).json(response);
      } else {
        const response: ApiResponse<null> = {
          success: false,
          error: {
            code: ERROR_CODES.AUTH_INVALID_TOKEN,
            message: 'Invalid authentication token'
          }
        };
        return res.status(401).json(response);
      }
    }
    
    // Pass other errors to error handler
    next(error);
  }
};