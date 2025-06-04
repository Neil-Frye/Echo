import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import { personalityRoutes } from './routes'
import { errorHandler } from './middleware/errorHandler'

const app = express()
const port = process.env.PORT || 3004

app.use(cors())
app.use(helmet())
app.use(express.json())

app.use('/api/v1/personality', personalityRoutes)

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'ai-personality-service' })
})

app.use(errorHandler)

app.listen(port, () => {
  console.log(`AI Personality service running on port ${port}`)
})