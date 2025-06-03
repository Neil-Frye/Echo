export function Features() {
  const features = [
    {
      title: 'AI-Powered Conversations',
      description: 'Our advanced AI learns your speech patterns, knowledge, and personality traits.',
      icon: '🤖'
    },
    {
      title: 'Memory Preservation',
      description: 'Upload photos, videos, and stories to enhance your digital consciousness.',
      icon: '🧠'
    },
    {
      title: 'Secure & Private',
      description: 'Your data is encrypted and securely stored. You control who can access your digital echo.',
      icon: '🔒'
    },
    {
      title: 'Cross-Platform',
      description: 'Access your digital consciousness via web, mobile, or smart speakers.',
      icon: '📱'
    }
  ]

  return (
    <section className="py-20 bg-slate-50">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl font-bold text-center mb-12">How It Works</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition">
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-slate-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}