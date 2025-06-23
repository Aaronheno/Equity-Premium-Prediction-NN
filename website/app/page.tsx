import Hero from '@/components/Hero'
import ResearchOverview from '@/components/ResearchOverview'
import ImplementationShowcase from '@/components/ImplementationShowcase'
import ResultsHighlights from '@/components/ResultsHighlights'

export default function Home() {
  return (
    <div className="min-h-screen">
      <Hero />
      <ResearchOverview />
      <ImplementationShowcase />
      <ResultsHighlights />
    </div>
  )
}