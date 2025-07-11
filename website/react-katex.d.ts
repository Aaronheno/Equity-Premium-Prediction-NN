declare module 'react-katex' {
  import { ComponentType } from 'react'

  interface MathProps {
    children?: string
    math?: string
    errorColor?: string
    renderError?: (error: Error) => JSX.Element
  }

  export const InlineMath: ComponentType<MathProps>
  export const BlockMath: ComponentType<MathProps>
}