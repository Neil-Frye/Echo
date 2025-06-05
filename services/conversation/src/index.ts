import express from 'express'
import { createServer } from 'http'
import { Server } from 'socket.io'
import cors from 'cors'
import { conversationRoutes } from './routes'
import { setupSocketHandlers } from './socket/handlers'
import { errorHandler } from './middleware/errorHandler'

const app = express()
const httpServer = createServer(app)
const io = new Server(httpServer, {
  cors: {
    origin: process.env.FRONTEND_URL || 'http://localhost:3000',
    credentials: true
  }
})

const port = process.env.PORT || 3005

app.use(cors())
app.use(express.json())

app.use('/api/v1/conversation', conversationRoutes)

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'conversation-service' })
})

app.use(errorHandler)

// Setup WebSocket handlers
setupSocketHandlers(io)

httpServer.listen(port, () => {
  console.log(`Conversation service running on port ${port}`)
})
