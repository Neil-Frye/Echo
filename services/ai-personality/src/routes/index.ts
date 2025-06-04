import { Router } from 'express'
import { personalityController } from '../controllers/personalityController'
import { authenticate } from '../middleware/authenticate'
import { validate } from '../middleware/validation'

const router = Router()

// Create/update personality model
router.post(
  '/train',
  authenticate,
  validate({
    body: ['contentIds'],
  }),
  personalityController.trainPersonalityModel
)

// Generate personality-based response
router.post(
  '/generate',
  authenticate,
  validate({
    body: ['message', 'conversationId'],
  }),
  personalityController.generateResponse
)

// Get personality profile
router.get(
  '/profile',
  authenticate,
  personalityController.getPersonalityProfile
)

export const personalityRoutes = router