import { redirect } from 'next/navigation'
import { getServerSession } from 'next-auth'
import { DashboardOverview } from '@/components/dashboard/DashboardOverview'
import { authOptions } from '@/lib/auth'

export default async function DashboardPage() {
  const session = await getServerSession(authOptions)
  
  if (!session) {
    redirect('/login')
  }

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
      <DashboardOverview userId={session.user.id} />
    </div>
  )
}