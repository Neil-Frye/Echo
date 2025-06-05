import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import { createProxyMiddleware } from 'http-proxy-middleware'
import rateLimit from 'express-rate-limit'

const app = express()
const port = process.env.PORT || 4000

// Security middleware
app.use(helmet())
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}))

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
})
app.use('/api/', limiter)

// Service routes
const services = {
  '/api/v1/auth': {
    target: 'http://localhost:3001',
    changeOrigin: true
  },
  '/api/v1/voice': {
    target: 'http://localhost:3002',
    changeOrigin: true
  },
  '/api/v1/content': {
    target: 'http://localhost:3003',
    changeOrigin: true
  },
  '/api/v1/personality': {
    target: 'http://localhost:3004',
    changeOrigin: true
  },
  '/api/v1/conversation': {
    target: 'http://localhost:3005',
    changeOrigin: true,
    ws: true // Enable WebSocket proxy
  },
  '/api/v1/billing': {
    target: 'http://localhost:3006',
    changeOrigin: true
  }
}

// Setup proxies
Object.entries(services).forEach(([path, config]) => {
  app.use(path, createProxyMiddleware(config))
})

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'api-gateway' })
})

app.listen(port, () => {
  console.log(`API Gateway running on port ${port}`)
})