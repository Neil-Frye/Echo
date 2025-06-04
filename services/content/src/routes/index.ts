import { Router } from 'express'
import multer from 'multer'
import { uploadContent, getUserContent } from '../controllers/contentController'
import { authenticate } from '../middleware/authenticate'
import { validate } from '../middleware/validation'

const router = Router()
const upload = multer({ storage: multer.memoryStorage() })

// Content upload route
router.post(
  '/upload',
  authenticate,
  upload.single('content'),
  validate({
    body: ['title'],
  }),
  uploadContent
)

// Get user content
router.get(
  '/',
  authenticate,
  getUserContent
)

export const contentRoutes = router