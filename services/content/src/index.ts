import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import morgan from 'morgan'
import { contentRoutes } from './routes'
import { errorHandler } from './middleware/errorHandler'

const app = express()
const port = process.env.PORT || 3003

app.use(cors())
app.use(helmet())
app.use(morgan('combined'))
app.use(express.json())

app.use('/api/v1/content', contentRoutes)

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'content-service' })
})

app.use(errorHandler)

app.listen(port, () => {
  console.log(`Content service running on port ${port}`)
})