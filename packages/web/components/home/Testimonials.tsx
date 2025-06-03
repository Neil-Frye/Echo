export function Testimonials() {
  const testimonials = [
    {
      quote: "EthernalEcho has given me peace of mind knowing my grandchildren will be able to hear my stories and advice long after I'm gone.",
      author: "Maria J.",
      role: "Legacy Plan User",
      image: "/testimonials/maria.jpg"
    },
    {
      quote: "The AI voice is incredibly accurate. My family was amazed at how it captured not just my words, but my tone and mannerisms.",
      author: "David T.",
      role: "Premium Plan User",
      image: "/testimonials/david.jpg"
    },
    {
      quote: "Setting up was easy, and within weeks my digital echo was already responding like me. This technology is truly remarkable.",
      author: "Sarah L.",
      role: "Basic Plan User",
      image: "/testimonials/sarah.jpg"
    }
  ]

  return (
    <section className="py-20 bg-slate-900 text-white">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl font-bold text-center mb-12">What Our Users Say</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="bg-slate-800 p-6 rounded-lg">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-slate-600 rounded-full mr-4">
                  {/* Replace with actual image: <img src={testimonial.image} alt={testimonial.author} className="w-12 h-12 rounded-full" /> */}
                </div>
                <div>
                  <h3 className="font-bold">{testimonial.author}</h3>
                  <p className="text-slate-400 text-sm">{testimonial.role}</p>
                </div>
              </div>
              <p className="text-slate-300 italic">"{testimonial.quote}"</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}