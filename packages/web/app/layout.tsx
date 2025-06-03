import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'EthernalEcho - Preserve Your Consciousness',
  description: 'EthernalEcho creates a digital consciousness that captures your essence, allowing you to leave a lasting legacy for future generations.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <header className="bg-slate-800 text-white">
          <div className="container mx-auto p-4 flex justify-between items-center">
            <div className="flex items-center">
              <a href="/" className="text-2xl font-bold">EthernalEcho</a>
            </div>
            <nav>
              <ul className="flex space-x-6">
                <li><a href="/" className="hover:text-blue-300">Home</a></li>
                <li><a href="/features" className="hover:text-blue-300">Features</a></li>
                <li><a href="/pricing" className="hover:text-blue-300">Pricing</a></li>
                <li><a href="/login" className="hover:text-blue-300">Login</a></li>
                <li><a href="/signup" className="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded">Sign Up</a></li>
              </ul>
            </nav>
          </div>
        </header>
        {children}
        <footer className="bg-slate-900 text-white py-12">
          <div className="container mx-auto px-6">
            <div className="grid md:grid-cols-4 gap-8">
              <div>
                <h3 className="text-xl font-bold mb-4">EthernalEcho</h3>
                <p className="text-slate-400">Preserving consciousness for future generations.</p>
              </div>
              <div>
                <h4 className="font-bold mb-4">Company</h4>
                <ul className="space-y-2">
                  <li><a href="/about" className="text-slate-400 hover:text-white">About Us</a></li>
                  <li><a href="/careers" className="text-slate-400 hover:text-white">Careers</a></li>
                  <li><a href="/blog" className="text-slate-400 hover:text-white">Blog</a></li>
                </ul>
              </div>
              <div>
                <h4 className="font-bold mb-4">Resources</h4>
                <ul className="space-y-2">
                  <li><a href="/docs" className="text-slate-400 hover:text-white">Documentation</a></li>
                  <li><a href="/help" className="text-slate-400 hover:text-white">Help Center</a></li>
                  <li><a href="/privacy" className="text-slate-400 hover:text-white">Privacy Policy</a></li>
                </ul>
              </div>
              <div>
                <h4 className="font-bold mb-4">Connect</h4>
                <ul className="space-y-2">
                  <li><a href="https://twitter.com" className="text-slate-400 hover:text-white">Twitter</a></li>
                  <li><a href="https://linkedin.com" className="text-slate-400 hover:text-white">LinkedIn</a></li>
                  <li><a href="/contact" className="text-slate-400 hover:text-white">Contact Us</a></li>
                </ul>
              </div>
            </div>
            <div className="border-t border-slate-800 mt-8 pt-8 text-center text-slate-400">
              <p>© {new Date().getFullYear()} EthernalEcho. All rights reserved.</p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}