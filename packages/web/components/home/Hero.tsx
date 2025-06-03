import Link from 'next/link'

export function Hero() {
  return (
    <section className="bg-gradient-to-r from-slate-900 to-blue-900 text-white py-24">
      <div className="container mx-auto px-6 text-center">
        <h1 className="text-5xl font-bold mb-6">Preserve Your Consciousness</h1>
        <p className="text-xl mb-12 max-w-2xl mx-auto">
          EthernalEcho creates a digital consciousness that captures your essence, allowing you to leave a lasting legacy for future generations.
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/signup" className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-lg transition">
            Get Started
          </Link>
          <Link href="/demo" className="bg-transparent border border-white hover:bg-white hover:text-blue-900 text-white font-bold py-3 px-6 rounded-lg transition">
            Try Demo
          </Link>
        </div>
      </div>
    </section>
  )
}