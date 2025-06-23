/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Dark Professional Theme
        'bg-primary': '#0a0b0f',
        'bg-secondary': '#1a1b23',
        'bg-tertiary': '#2a2d3a',
        'accent-blue': '#3b82f6',
        'accent-purple': '#8b5cf6',
        'accent-green': '#10b981',
        'accent-orange': '#f59e0b',
        'accent-red': '#ef4444',
        'text-primary': '#f8fafc',
        'text-secondary': '#cbd5e1',
        'text-muted': '#64748b',
        'code-bg': '#1e293b',
        'code-border': '#334155',
        'formula-bg': '#0f172a',
        'formula-accent': '#06b6d4',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'Consolas', 'monospace'],
        'math': ['Computer Modern', 'serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}