import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import 'katex/dist/katex.min.css'
import Navigation from '@/components/Navigation'
import Footer from '@/components/Footer'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  metadataBase: new URL('https://neural-networks-equity-prediction.vercel.app'),
  title: 'Neural Networks for Equity Premium Prediction',
  description: 'Comprehensive documentation and interactive guide for neural network architectures in financial prediction, featuring 8 model types and hyperparameter optimization strategies.',
  keywords: [
    'neural networks',
    'equity premium prediction',
    'financial machine learning',
    'hyperparameter optimization',
    'pytorch',
    'time series forecasting'
  ],
  authors: [{ name: 'Aaron Hennessy' }],
  creator: 'Aaron Hennessy',
  openGraph: {
    title: 'Neural Networks for Equity Premium Prediction',
    description: 'Interactive documentation for neural network architectures in financial prediction',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Neural Networks for Equity Premium Prediction',
    description: 'Interactive documentation for neural network architectures in financial prediction',
  },
  robots: 'index, follow',
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-bg-primary text-text-primary">
        <Navigation />
        <div className="flex min-h-screen">
          {/* Main content area - offset by sidebar on desktop and top nav */}
          <main className="flex-1 lg:ml-80 pt-16">
            {children}
          </main>
        </div>
        <Footer />
      </body>
    </html>
  )
}