import express from 'express'
import cors from 'cors'
import Stripe from 'stripe'
import { billingRoutes } from './routes'
import { webhookHandler } from './controllers/webhookController'
import { errorHandler } from './middleware/errorHandler'

const app = express()
const port = process.env.PORT || 3006

// Stripe webhook needs raw body
app.post(
  '/webhook/stripe',
  express.raw({ type: 'application/json' }),
  webhookHandler
)

app.use(cors())
app.use(express.json())

app.use('/api/v1/billing', billingRoutes)

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'billing-service' })
})

app.use(errorHandler)

app.listen(port, () => {
  console.log(`Billing service running on port ${port}`)
})