import { useState } from 'react'

interface DashboardOverviewProps {
  userId: string
}

export function DashboardOverview({ userId }: DashboardOverviewProps) {
  const [isLoading, setIsLoading] = useState(false)
  
  // This would normally fetch real data
  const stats = [
    { label: 'Training Hours', value: '12.5' },
    { label: 'Memory Items', value: '342' },
    { label: 'AI Accuracy', value: '87%' },
    { label: 'Storage Used', value: '2.4GB' }
  ]

  const recentConversations = [
    { id: 1, date: '2023-06-01', length: '5m 23s', topic: 'Childhood memories' },
    { id: 2, date: '2023-05-28', length: '12m 7s', topic: 'Career advice' },
    { id: 3, date: '2023-05-25', length: '3m 45s', topic: 'Family recipes' }
  ]

  return (
    <div>
      <div className="grid md:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow">
            <p className="text-sm text-slate-500 mb-1">{stat.label}</p>
            <p className="text-2xl font-bold">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="bg-white p-6 rounded-lg shadow mb-8">
        <h2 className="text-xl font-bold mb-4">Recent Training Sessions</h2>
        {isLoading ? (
          <p>Loading...</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b">
                  <th className="py-2 px-4 text-left">Date</th>
                  <th className="py-2 px-4 text-left">Length</th>
                  <th className="py-2 px-4 text-left">Topic</th>
                  <th className="py-2 px-4 text-left">Actions</th>
                </tr>
              </thead>
              <tbody>
                {recentConversations.map((conversation) => (
                  <tr key={conversation.id} className="border-b hover:bg-slate-50">
                    <td className="py-2 px-4">{conversation.date}</td>
                    <td className="py-2 px-4">{conversation.length}</td>
                    <td className="py-2 px-4">{conversation.topic}</td>
                    <td className="py-2 px-4">
                      <button className="text-blue-500 hover:text-blue-700 mr-2">View</button>
                      <button className="text-red-500 hover:text-red-700">Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Upload Memory</h2>
          <p className="mb-4 text-slate-600">Add photos, videos, or documents to enhance your digital consciousness.</p>
          <button 
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            onClick={() => alert('Upload functionality would open here')}
          >
            Upload Files
          </button>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-bold mb-4">Schedule Training</h2>
          <p className="mb-4 text-slate-600">Regular training sessions improve your digital echo's accuracy.</p>
          <button 
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded"
            onClick={() => alert('Scheduling modal would open here')}
          >
            Schedule Session
          </button>
        </div>
      </div>
    </div>
  )
}