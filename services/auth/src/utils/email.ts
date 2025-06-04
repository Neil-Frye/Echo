import nodemailer from 'nodemailer';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Email configuration
const emailConfig = {
  host: process.env.EMAIL_HOST || 'smtp.example.com',
  port: parseInt(process.env.EMAIL_PORT || '587'),
  secure: process.env.EMAIL_SECURE === 'true',
  auth: {
    user: process.env.EMAIL_USER || 'user@example.com',
    pass: process.env.EMAIL_PASSWORD || 'password',
  },
};

// Frontend URLs
const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:3000';

// Create transporter
const transporter = nodemailer.createTransport(emailConfig);

/**
 * Send verification email
 * @param email Recipient email
 * @param token Verification token
 */
export const sendVerificationEmail = async (email: string, token: string) => {
  const verificationUrl = `${frontendUrl}/verify-email/${token}`;
  
  const mailOptions = {
    from: `"EthernalEcho" <${emailConfig.auth.user}>`,
    to: email,
    subject: 'Verify Your EthernalEcho Account',
    text: `
      Welcome to EthernalEcho!
      
      Please verify your email address by clicking the link below:
      
      ${verificationUrl}
      
      This link will expire in 24 hours.
      
      If you did not create an account, please ignore this email.
      
      Best regards,
      The EthernalEcho Team
    `,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2>Welcome to EthernalEcho!</h2>
        <p>Thank you for creating an account. Please verify your email address by clicking the button below:</p>
        <p style="text-align: center;">
          <a href="${verificationUrl}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Verify Email
          </a>
        </p>
        <p>Or copy and paste this link into your browser:</p>
        <p>${verificationUrl}</p>
        <p>This link will expire in 24 hours.</p>
        <p>If you did not create an account, please ignore this email.</p>
        <p>Best regards,<br>The EthernalEcho Team</p>
      </div>
    `,
  };
  
  try {
    // In development, log email instead of sending
    if (process.env.NODE_ENV === 'development') {
      console.log('Verification Email:', mailOptions);
      return;
    }
    
    await transporter.sendMail(mailOptions);
  } catch (error) {
    console.error('Error sending verification email:', error);
    throw new Error('Failed to send verification email');
  }
};

/**
 * Send password reset email
 * @param email Recipient email
 * @param token Reset token
 */
export const sendPasswordResetEmail = async (email: string, token: string) => {
  const resetUrl = `${frontendUrl}/reset-password/${token}`;
  
  const mailOptions = {
    from: `"EthernalEcho" <${emailConfig.auth.user}>`,
    to: email,
    subject: 'Reset Your EthernalEcho Password',
    text: `
      You requested a password reset for your EthernalEcho account.
      
      Please click the link below to reset your password:
      
      ${resetUrl}
      
      This link will expire in 1 hour.
      
      If you did not request a password reset, please ignore this email.
      
      Best regards,
      The EthernalEcho Team
    `,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2>Reset Your Password</h2>
        <p>You requested a password reset for your EthernalEcho account. Please click the button below to reset your password:</p>
        <p style="text-align: center;">
          <a href="${resetUrl}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
            Reset Password
          </a>
        </p>
        <p>Or copy and paste this link into your browser:</p>
        <p>${resetUrl}</p>
        <p>This link will expire in 1 hour.</p>
        <p>If you did not request a password reset, please ignore this email.</p>
        <p>Best regards,<br>The EthernalEcho Team</p>
      </div>
    `,
  };
  
  try {
    // In development, log email instead of sending
    if (process.env.NODE_ENV === 'development') {
      console.log('Password Reset Email:', mailOptions);
      return;
    }
    
    await transporter.sendMail(mailOptions);
  } catch (error) {
    console.error('Error sending password reset email:', error);
    throw new Error('Failed to send password reset email');
  }
};