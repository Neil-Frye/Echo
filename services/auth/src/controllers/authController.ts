import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import { 
  UserStatus, 
  ApiResponse, 
  ERROR_CODES 
} from '@ethernalecho/shared';
import { db } from '../db';
import { sendVerificationEmail, sendPasswordResetEmail } from '../utils/email';
import { generateTwoFactorSecret, verifyTwoFactorToken } from '../utils/twoFactor';

// JWT configuration
const JWT_SECRET = process.env.JWT_SECRET || 'default-jwt-secret';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1h';
const JWT_REFRESH_SECRET = process.env.JWT_REFRESH_SECRET || 'default-refresh-secret';
const JWT_REFRESH_EXPIRES_IN = process.env.JWT_REFRESH_EXPIRES_IN || '7d';

/**
 * Register a new user
 */
export const register = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email, password, fullName } = req.body;
    
    // Check if user already exists
    const existingUser = await db.query(
      'SELECT * FROM users WHERE email = $1',
      [email]
    );
    
    if (existingUser.rows.length > 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_ALREADY_EXISTS,
          message: 'User with this email already exists'
        }
      };
      return res.status(409).json(response);
    }
    
    // Hash password
    const saltRounds = 10;
    const passwordHash = await bcrypt.hash(password, saltRounds);
    
    // Create user
    const userId = uuidv4();
    await db.query(
      `INSERT INTO users (
        id, email, password_hash, status, created_at, updated_at
      ) VALUES ($1, $2, $3, $4, NOW(), NOW())`,
      [userId, email, passwordHash, UserStatus.Active]
    );
    
    // Create profile
    await db.query(
      `INSERT INTO profiles (
        id, user_id, full_name, created_at, updated_at
      ) VALUES ($1, $2, $3, NOW(), NOW())`,
      [uuidv4(), userId, fullName]
    );
    
    // Generate verification token
    const verificationToken = jwt.sign(
      { userId },
      JWT_SECRET,
      { expiresIn: '24h' }
    );
    
    // Send verification email
    await sendVerificationEmail(email, verificationToken);
    
    const response: ApiResponse<{ userId: string }> = {
      success: true,
      data: { userId }
    };
    
    res.status(201).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Login user
 */
export const login = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email, password } = req.body;
    
    // Find user
    const result = await db.query(
      'SELECT * FROM users WHERE email = $1',
      [email]
    );
    
    const user = result.rows[0];
    
    // Check if user exists
    if (!user) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_CREDENTIALS,
          message: 'Invalid email or password'
        }
      };
      return res.status(401).json(response);
    }
    
    // Check if account is active
    if (user.status !== UserStatus.Active) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_ACCOUNT_DISABLED,
          message: 'Account is not active'
        }
      };
      return res.status(403).json(response);
    }
    
    // Verify password
    const passwordMatch = await bcrypt.compare(password, user.password_hash);
    if (!passwordMatch) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_CREDENTIALS,
          message: 'Invalid email or password'
        }
      };
      return res.status(401).json(response);
    }
    
    // Check if email is verified
    if (!user.email_verified) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_EMAIL_NOT_VERIFIED,
          message: 'Email not verified'
        }
      };
      return res.status(403).json(response);
    }
    
    // Check if 2FA is enabled
    if (user.two_factor_enabled) {
      const response: ApiResponse<{ userId: string, requiresTwoFactor: boolean }> = {
        success: true,
        data: {
          userId: user.id,
          requiresTwoFactor: true
        }
      };
      return res.status(200).json(response);
    }
    
    // Generate tokens
    const accessToken = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN }
    );
    
    const refreshToken = jwt.sign(
      { userId: user.id },
      JWT_REFRESH_SECRET,
      { expiresIn: JWT_REFRESH_EXPIRES_IN }
    );
    
    const response: ApiResponse<{
      userId: string;
      accessToken: string;
      refreshToken: string;
    }> = {
      success: true,
      data: {
        userId: user.id,
        accessToken,
        refreshToken
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Verify email
 */
export const verifyEmail = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { token } = req.params;
    
    // Verify token
    let decoded;
    try {
      decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
    } catch (error) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid or expired token'
        }
      };
      return res.status(400).json(response);
    }
    
    // Update user
    const result = await db.query(
      'UPDATE users SET email_verified = TRUE, updated_at = NOW() WHERE id = $1 RETURNING *',
      [decoded.userId]
    );
    
    if (result.rows.length === 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_NOT_FOUND,
          message: 'User not found'
        }
      };
      return res.status(404).json(response);
    }
    
    const response: ApiResponse<{ verified: boolean }> = {
      success: true,
      data: {
        verified: true
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Forgot password
 */
export const forgotPassword = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { email } = req.body;
    
    // Find user
    const result = await db.query(
      'SELECT * FROM users WHERE email = $1',
      [email]
    );
    
    // Always return success even if user doesn't exist (security)
    if (result.rows.length === 0) {
      const response: ApiResponse<{ emailSent: boolean }> = {
        success: true,
        data: {
          emailSent: true
        }
      };
      return res.status(200).json(response);
    }
    
    const user = result.rows[0];
    
    // Generate reset token
    const resetToken = jwt.sign(
      { userId: user.id },
      JWT_SECRET,
      { expiresIn: '1h' }
    );
    
    // Send password reset email
    await sendPasswordResetEmail(email, resetToken);
    
    const response: ApiResponse<{ emailSent: boolean }> = {
      success: true,
      data: {
        emailSent: true
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Reset password
 */
export const resetPassword = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { token } = req.params;
    const { password } = req.body;
    
    // Verify token
    let decoded;
    try {
      decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
    } catch (error) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid or expired token'
        }
      };
      return res.status(400).json(response);
    }
    
    // Hash new password
    const saltRounds = 10;
    const passwordHash = await bcrypt.hash(password, saltRounds);
    
    // Update user
    const result = await db.query(
      'UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2 RETURNING *',
      [passwordHash, decoded.userId]
    );
    
    if (result.rows.length === 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_NOT_FOUND,
          message: 'User not found'
        }
      };
      return res.status(404).json(response);
    }
    
    const response: ApiResponse<{ reset: boolean }> = {
      success: true,
      data: {
        reset: true
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Refresh token
 */
export const refreshToken = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { refreshToken } = req.body;
    
    if (!refreshToken) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_MISSING_TOKEN,
          message: 'Refresh token is required'
        }
      };
      return res.status(400).json(response);
    }
    
    // Verify refresh token
    let decoded;
    try {
      decoded = jwt.verify(refreshToken, JWT_REFRESH_SECRET) as { userId: string };
    } catch (error) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid or expired refresh token'
        }
      };
      return res.status(401).json(response);
    }
    
    // Get user
    const result = await db.query(
      'SELECT * FROM users WHERE id = $1',
      [decoded.userId]
    );
    
    if (result.rows.length === 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_NOT_FOUND,
          message: 'User not found'
        }
      };
      return res.status(404).json(response);
    }
    
    const user = result.rows[0];
    
    // Generate new access token
    const accessToken = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN }
    );
    
    // Generate new refresh token
    const newRefreshToken = jwt.sign(
      { userId: user.id },
      JWT_REFRESH_SECRET,
      { expiresIn: JWT_REFRESH_EXPIRES_IN }
    );
    
    const response: ApiResponse<{
      accessToken: string;
      refreshToken: string;
    }> = {
      success: true,
      data: {
        accessToken,
        refreshToken: newRefreshToken
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Logout
 */
export const logout = async (req: Request, res: Response, next: NextFunction) => {
  try {
    // In a real implementation, you might want to invalidate the token
    // by adding it to a blacklist or database of revoked tokens
    
    const response: ApiResponse<{ loggedOut: boolean }> = {
      success: true,
      data: {
        loggedOut: true
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Setup two-factor authentication
 */
export const setupTwoFactor = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { userId } = req.user as { userId: string };
    
    // Generate 2FA secret
    const { secret, qrCodeUrl } = generateTwoFactorSecret();
    
    // Store secret in database
    await db.query(
      'UPDATE users SET two_factor_secret = $1, updated_at = NOW() WHERE id = $2',
      [secret, userId]
    );
    
    const response: ApiResponse<{
      qrCodeUrl: string;
    }> = {
      success: true,
      data: {
        qrCodeUrl
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Verify two-factor authentication
 */
export const verifyTwoFactor = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { userId } = req.user as { userId: string };
    const { token, enable } = req.body;
    
    // Get user
    const result = await db.query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );
    
    if (result.rows.length === 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_NOT_FOUND,
          message: 'User not found'
        }
      };
      return res.status(404).json(response);
    }
    
    const user = result.rows[0];
    
    // Verify token
    const isValid = verifyTwoFactorToken(token, user.two_factor_secret);
    
    if (!isValid) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid two-factor token'
        }
      };
      return res.status(401).json(response);
    }
    
    if (enable) {
      // Enable 2FA
      await db.query(
        'UPDATE users SET two_factor_enabled = TRUE, updated_at = NOW() WHERE id = $1',
        [userId]
      );
    }
    
    // Generate tokens
    const accessToken = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: JWT_EXPIRES_IN }
    );
    
    const refreshToken = jwt.sign(
      { userId: user.id },
      JWT_REFRESH_SECRET,
      { expiresIn: JWT_REFRESH_EXPIRES_IN }
    );
    
    const response: ApiResponse<{
      verified: boolean;
      accessToken: string;
      refreshToken: string;
    }> = {
      success: true,
      data: {
        verified: true,
        accessToken,
        refreshToken
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};

/**
 * Disable two-factor authentication
 */
export const disableTwoFactor = async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { userId } = req.user as { userId: string };
    const { token } = req.body;
    
    // Get user
    const result = await db.query(
      'SELECT * FROM users WHERE id = $1',
      [userId]
    );
    
    if (result.rows.length === 0) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.USER_NOT_FOUND,
          message: 'User not found'
        }
      };
      return res.status(404).json(response);
    }
    
    const user = result.rows[0];
    
    // Verify token
    const isValid = verifyTwoFactorToken(token, user.two_factor_secret);
    
    if (!isValid) {
      const response: ApiResponse<null> = {
        success: false,
        error: {
          code: ERROR_CODES.AUTH_INVALID_TOKEN,
          message: 'Invalid two-factor token'
        }
      };
      return res.status(401).json(response);
    }
    
    // Disable 2FA
    await db.query(
      'UPDATE users SET two_factor_enabled = FALSE, two_factor_secret = NULL, updated_at = NOW() WHERE id = $1',
      [userId]
    );
    
    const response: ApiResponse<{ disabled: boolean }> = {
      success: true,
      data: {
        disabled: true
      }
    };
    
    res.status(200).json(response);
  } catch (error) {
    next(error);
  }
};