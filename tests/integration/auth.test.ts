import request from 'supertest'
import app from '../../services/auth/src/index'
import { db } from '../../services/auth/src/db'

describe('Authentication Flow', () => {
  beforeEach(async () => {
    await db.query('DELETE FROM users')
  })

  afterAll(async () => {
    await db.end()
  })

  test('Complete authentication flow', async () => {
    // Register
    const registerRes = await request(app)
      .post('/api/v1/auth/register')
      .send({
        email: 'test@example.com',
        password: 'SecurePass123!',
        fullName: 'Test User'
      })

    expect(registerRes.status).toBe(201)
    expect(registerRes.body.token).toBeDefined()

    // Login
    const loginRes = await request(app)
      .post('/api/v1/auth/login')
      .send({
        email: 'test@example.com',
        password: 'SecurePass123!'
      })

    expect(loginRes.status).toBe(200)
    expect(loginRes.body.token).toBeDefined()

    // Verify token
    const token = loginRes.body.token
    const profileRes = await request(app)
      .get('/api/v1/auth/profile')
      .set('Authorization', `Bearer ${token}`)

    expect(profileRes.status).toBe(200)
    expect(profileRes.body.email).toBe('test@example.com')
  })
})