'use client'

import Link from 'next/link'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { motion } from 'framer-motion'

interface NavigationButtonsProps {
  prevHref?: string
  prevLabel?: string
  nextHref?: string
  nextLabel?: string
}

export default function NavigationButtons({
  prevHref,
  prevLabel,
  nextHref,
  nextLabel
}: NavigationButtonsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="flex justify-between items-center pt-12 border-t border-bg-tertiary"
    >
      {/* Previous Page */}
      <div className="flex-1">
        {prevHref && prevLabel ? (
          <Link
            href={prevHref}
            className="group inline-flex items-center space-x-3 text-text-secondary hover:text-accent-blue transition-colors"
          >
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-bg-secondary border border-bg-tertiary group-hover:border-accent-blue/30 group-hover:bg-accent-blue/10 transition-all">
              <ChevronLeft className="w-5 h-5" />
            </div>
            <div className="text-left">
              <div className="text-xs text-text-muted uppercase tracking-wide">Previous</div>
              <div className="font-medium">{prevLabel}</div>
            </div>
          </Link>
        ) : (
          <div></div> // Empty div to maintain flex spacing
        )}
      </div>

      {/* Center spacer */}
      <div className="flex-1 flex justify-center">
        {/* Dots removed as they were non-functional decorative elements */}
      </div>

      {/* Next Page */}
      <div className="flex-1 flex justify-end">
        {nextHref && nextLabel ? (
          <Link
            href={nextHref}
            className="group inline-flex items-center space-x-3 text-text-secondary hover:text-accent-blue transition-colors"
          >
            <div className="text-right">
              <div className="text-xs text-text-muted uppercase tracking-wide">Next</div>
              <div className="font-medium">{nextLabel}</div>
            </div>
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-bg-secondary border border-bg-tertiary group-hover:border-accent-blue/30 group-hover:bg-accent-blue/10 transition-all">
              <ChevronRight className="w-5 h-5" />
            </div>
          </Link>
        ) : (
          <div></div> // Empty div to maintain flex spacing
        )}
      </div>
    </motion.div>
  )
}