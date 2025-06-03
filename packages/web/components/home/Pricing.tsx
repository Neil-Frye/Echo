import Link from 'next/link'

export function Pricing() {
  const plans = [
    {
      name: 'Basic',
      price: '$9.99',
      period: 'per month',
      description: 'Start your digital consciousness journey',
      features: [
        '5 hours of voice training',
        'Basic memory storage (5GB)',
        'Web access',
        'Monthly updates'
      ],
      cta: 'Get Started',
      highlighted: false
    },
    {
      name: 'Premium',
      price: '$24.99',
      period: 'per month',
      description: 'Enhanced features for a more authentic echo',
      features: [
        'Unlimited voice training',
        'Advanced memory storage (50GB)',
        'Web & mobile access',
        'Weekly updates',
        'Priority support'
      ],
      cta: 'Go Premium',
      highlighted: true
    },
    {
      name: 'Legacy',
      price: '$99.99',
      period: 'per month',
      description: 'The ultimate digital consciousness experience',
      features: [
        'Unlimited voice training',
        'Infinite memory storage',
        'All platform access',
        'Daily updates',
        '24/7 support',
        'Family access controls'
      ],
      cta: 'Start Legacy',
      highlighted: false
    }
  ]

  return (
    <section className="py-20">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl font-bold text-center mb-12">Pricing Plans</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <div 
              key={index} 
              className={`rounded-lg overflow-hidden shadow-lg ${plan.highlighted ? 'ring-2 ring-blue-500 scale-105' : ''}`}
            >
              <div className={`p-6 ${plan.highlighted ? 'bg-blue-500 text-white' : 'bg-white'}`}>
                <h3 className="text-2xl font-bold mb-1">{plan.name}</h3>
                <div className="mb-4">
                  <span className="text-3xl font-bold">{plan.price}</span>
                  <span className="text-sm ml-1">{plan.period}</span>
                </div>
                <p className={`mb-6 ${plan.highlighted ? 'text-blue-100' : 'text-slate-600'}`}>{plan.description}</p>
              </div>
              <div className="p-6 bg-white">
                <ul className="mb-6 space-y-2">
                  {plan.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center">
                      <svg className="h-5 w-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
                <Link 
                  href="/signup" 
                  className={`block text-center py-2 px-4 rounded w-full font-bold ${plan.highlighted
                    ? 'bg-blue-500 hover:bg-blue-600 text-white'
                    : 'bg-slate-800 hover:bg-slate-900 text-white'}`}
                >
                  {plan.cta}
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}