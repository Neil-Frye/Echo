import { Request, Response, NextFunction } from 'express';
import { isValidEmail } from '@ethernalecho/shared';
import { ApiResponse, ERROR_CODES } from '@ethernalecho/shared';

/**
 * Validate login request
 */
export const validateLogin = (req: Request, res: Response, next: NextFunction) => {
  const { email, password } = req.body;
  
  // Validate email
  if (!email || !isValidEmail(email)) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Invalid email address'
      }
    };
    return res.status(400).json(response);
  }
  
  // Validate password
  if (!password || typeof password !== 'string' || password.length < 8) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Password is required and must be at least 8 characters'
      }
    };
    return res.status(400).json(response);
  }
  
  next();
};

/**
 * Validate register request
 */
export const validateRegister = (req: Request, res: Response, next: NextFunction) => {
  const { email, password, fullName } = req.body;
  
  // Validate email
  if (!email || !isValidEmail(email)) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Invalid email address'
      }
    };
    return res.status(400).json(response);
  }
  
  // Validate password
  if (!password || typeof password !== 'string' || password.length < 8) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Password is required and must be at least 8 characters'
      }
    };
    return res.status(400).json(response);
  }
  
  // Validate full name
  if (!fullName || typeof fullName !== 'string' || fullName.trim().length === 0) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Full name is required'
      }
    };
    return res.status(400).json(response);
  }
  
  next();
};

/**
 * Validate reset password request
 */
export const validateResetPassword = (req: Request, res: Response, next: NextFunction) => {
  const { password, confirmPassword } = req.body;
  
  // Validate password
  if (!password || typeof password !== 'string' || password.length < 8) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Password is required and must be at least 8 characters'
      }
    };
    return res.status(400).json(response);
  }
  
  // Validate password confirmation
  if (password !== confirmPassword) {
    const response: ApiResponse<null> = {
      success: false,
      error: {
        code: ERROR_CODES.USER_INVALID_INPUT,
        message: 'Passwords do not match'
      }
    };
    return res.status(400).json(response);
  }
  
  next();
};