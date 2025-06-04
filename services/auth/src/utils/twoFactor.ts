import speakeasy from 'speakeasy';
import QRCode from 'qrcode';

/**
 * Generate two-factor secret and QR code URL
 * @returns Two-factor secret and QR code URL
 */
export const generateTwoFactorSecret = () => {
  // Generate secret
  const secret = speakeasy.generateSecret({
    name: 'EthernalEcho',
    length: 20,
  });
  
  // Generate QR code URL
  const qrCodeUrl = QRCode.toDataURL(secret.otpauth_url);
  
  return {
    secret: secret.base32,
    qrCodeUrl,
  };
};

/**
 * Verify two-factor token
 * @param token Two-factor token
 * @param secret Two-factor secret
 * @returns Whether token is valid
 */
export const verifyTwoFactorToken = (token: string, secret: string) => {
  return speakeasy.totp.verify({
    secret,
    encoding: 'base32',
    token,
    window: 1, // Allow 30 seconds of clock drift
  });
};