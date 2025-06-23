'use client'

import { useState } from 'react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

interface MathFormulaProps {
  latex: string
  block?: boolean
  explanation?: string
  className?: string
}

export default function MathFormula({ 
  latex, 
  block = false, 
  explanation,
  className = ''
}: MathFormulaProps) {
  const [showExplanation, setShowExplanation] = useState(false)

  // For debugging - add fallback text
  if (!latex || latex.trim() === '') {
    return <span className="text-red-500">Empty formula</span>
  }

  if (block) {
    return (
      <div className={`math-container block rounded-lg p-6 my-6 ${className}`}>
        <div className="text-center">
          <BlockMath math={latex} />
        </div>
        {explanation && (
          <div className="mt-4">
            <button
              onClick={() => setShowExplanation(!showExplanation)}
              className="text-accent-blue text-sm hover:underline"
            >
              {showExplanation ? 'Hide explanation' : 'Show explanation'}
            </button>
            {showExplanation && (
              <div className="mt-2 p-4 bg-bg-secondary rounded-lg border border-bg-tertiary">
                <p className="text-text-secondary text-sm">{explanation}</p>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <span className={`math-container inline rounded px-2 py-1 ${className}`}>
      <InlineMath math={latex} />
    </span>
  )
}