import { Router } from 'express';
import { 
  login, 
  register, 
  verifyEmail, 
  forgotPassword, 
  resetPassword, 
  refreshToken, 
  logout,
  setupTwoFactor,
  verifyTwoFactor,
  disableTwoFactor
} from '../controllers/authController';
import { validateLogin, validateRegister, validateResetPassword } from '../middleware/validation';
import { authenticate } from '../middleware/authenticate';

const router = Router();

// Public routes
router.post('/register', validateRegister, register);
router.post('/login', validateLogin, login);
router.post('/verify-email/:token', verifyEmail);
router.post('/forgot-password', forgotPassword);
router.post('/reset-password/:token', validateResetPassword, resetPassword);
router.post('/refresh-token', refreshToken);

// Protected routes (require authentication)
router.post('/logout', authenticate, logout);
router.post('/2fa/setup', authenticate, setupTwoFactor);
router.post('/2fa/verify', authenticate, verifyTwoFactor);
router.post('/2fa/disable', authenticate, disableTwoFactor);

export const authRoutes = router;