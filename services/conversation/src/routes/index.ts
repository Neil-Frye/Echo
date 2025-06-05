import { Router } from 'express'
import { getConversation, createConversation } from '../controllers/conversationController'

const router = Router()

router.get('/:id', getConversation)
router.post('/', createConversation)

export { router as conversationRoutes }
