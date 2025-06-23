'use client'

import { LazyMotion, domAnimation } from 'framer-motion'
import { ReactNode } from 'react'

interface LazyMotionWrapperProps {
  children: ReactNode
}

export default function LazyMotionWrapper({ children }: LazyMotionWrapperProps) {
  return (
    <LazyMotion features={domAnimation} strict>
      {children}
    </LazyMotion>
  )
}