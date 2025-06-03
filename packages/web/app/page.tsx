import { Hero } from '@/components/home/Hero'
import { Features } from '@/components/home/Features'
import { Pricing } from '@/components/home/Pricing'
import { Testimonials } from '@/components/home/Testimonials'

export default function HomePage() {
  return (
    <main>
      <Hero />
      <Features />
      <Pricing />
      <Testimonials />
    </main>
  )
}