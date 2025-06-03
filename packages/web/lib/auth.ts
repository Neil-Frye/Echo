import { NextAuthOptions } from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' }
      },
      async authorize(credentials) {
        // This is a placeholder - in a real app, you would validate against your API
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // TODO: Replace with actual API call to validate credentials
        // For demo purposes, we'll accept a test user
        if (credentials.email === 'user@example.com' && credentials.password === 'password') {
          return {
            id: '1',
            name: 'Demo User',
            email: 'user@example.com'
          }
        }

        return null
      }
    })
  ],
  session: {
    strategy: 'jwt'
  },
  pages: {
    signIn: '/login',
    signOut: '/',
    error: '/login'
  },
  callbacks: {
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.sub || ''
      }
      return session
    }
  },
  secret: process.env.NEXTAUTH_SECRET || 'your-development-secret-do-not-use-in-production',
  debug: process.env.NODE_ENV === 'development'
}