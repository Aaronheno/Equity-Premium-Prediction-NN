'use client'

import { useState, memo } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check, Code2, Info, FileCode } from 'lucide-react'

interface CodeBlockProps {
  code: string
  language: string
  title?: string
  showLineNumbers?: boolean
  highlightLines?: number[]
  codeType?: 'conceptual' | 'actual' | 'simplified'
  actualImplementationPath?: string
}

function CodeBlock({ 
  code, 
  language, 
  title, 
  showLineNumbers = true, 
  highlightLines = [],
  codeType = 'actual',
  actualImplementationPath
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy code:', err)
    }
  }

  const customStyle = {
    ...oneDark,
    'pre[class*="language-"]': {
      ...oneDark['pre[class*="language-"]'],
      background: 'var(--code-bg)',
      margin: 0,
      padding: '1rem',
      fontSize: '0.875rem',
      lineHeight: '1.6',
    },
    'code[class*="language-"]': {
      ...oneDark['code[class*="language-"]'],
      background: 'transparent',
      fontSize: '0.875rem',
      fontFamily: 'var(--font-jetbrains-mono), Consolas, monospace',
    },
  }

  return (
    <div className="code-container my-6">
      {/* Code Type Notice */}
      {codeType !== 'actual' && (
        <div className={`code-notice ${
          codeType === 'conceptual' ? 'bg-accent-orange/10 border-accent-orange/20' : 'bg-accent-blue/10 border-accent-blue/20'
        } border rounded-t-lg p-3 text-sm`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Info className="w-4 h-4" />
              <span className="font-medium">
                {codeType === 'conceptual' 
                  ? 'Conceptual Example for Educational Clarity' 
                  : 'Simplified from Actual Implementation'}
              </span>
            </div>
            {actualImplementationPath && (
              <a 
                href={`https://github.com/your-repo/blob/main/${actualImplementationPath}`} 
                className="flex items-center space-x-1 text-accent-blue hover:underline text-xs"
                target="_blank"
                rel="noopener noreferrer"
              >
                <FileCode className="w-3 h-3" />
                <span>View actual: {actualImplementationPath}</span>
              </a>
            )}
          </div>
          <p className="text-text-muted text-xs mt-1">
            {codeType === 'conceptual' 
              ? 'This example demonstrates the underlying concepts. The actual implementation uses modular architecture for maintainability and reusability.'
              : 'This is a simplified version. The production code includes additional error handling, optimization, and configuration options.'}
          </p>
        </div>
      )}
      
      {/* Main Code Block */}
      <div className={`bg-code-bg border border-code-border ${
        codeType !== 'actual' ? 'rounded-b-lg' : 'rounded-lg'
      } overflow-hidden`}>
        {/* Header */}
        <div className="code-header bg-bg-tertiary px-4 py-3 border-b border-code-border flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {/* Traffic lights */}
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-accent-red"></div>
            <div className="w-3 h-3 rounded-full bg-accent-orange"></div>
            <div className="w-3 h-3 rounded-full bg-accent-green"></div>
          </div>
          
          {/* Language tag and title */}
          <div className="flex items-center space-x-3">
            <span className="language-tag bg-accent-blue text-white px-2 py-1 rounded text-xs font-medium uppercase">
              {language}
            </span>
            {title && (
              <>
                <Code2 className="w-4 h-4 text-text-muted" />
                <span className="text-sm text-text-secondary font-mono">{title}</span>
              </>
            )}
          </div>
        </div>

        {/* Copy button */}
        <button
          onClick={handleCopy}
          className="flex items-center space-x-2 px-3 py-1.5 bg-bg-secondary hover:bg-accent-blue/10 border border-bg-tertiary hover:border-accent-blue/30 rounded-md transition-all text-sm text-text-secondary hover:text-accent-blue"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-4 h-4" />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-4 h-4" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={customStyle}
          showLineNumbers={showLineNumbers}
          lineNumberStyle={{
            color: 'var(--text-muted)',
            backgroundColor: 'transparent',
            paddingRight: '1rem',
            minWidth: '3rem',
            textAlign: 'right',
            userSelect: 'none',
          }}
          wrapLines={true}
          lineProps={(lineNumber) => {
            const isHighlighted = highlightLines.includes(lineNumber)
            return {
              style: {
                backgroundColor: isHighlighted ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                display: 'block',
                width: '100%',
              }
            }
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
      </div>
    </div>
  )
}

export default memo(CodeBlock)