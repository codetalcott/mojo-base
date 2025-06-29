// uno.config.ts
// UnoCSS configuration for Mojo Semantic Search
// Ready for integration with your Elysiajs framework

import { defineConfig, presetUno, presetAttributify, presetTypography } from 'unocss'
import { presetDaisy } from 'unocss-preset-daisy'

export default defineConfig({
  presets: [
    presetUno(),
    presetAttributify(),
    presetTypography(),
    presetDaisy({
      themes: ['dark', 'light', 'cyberpunk']
    })
  ],
  
  // Custom theme for Mojo Semantic Search
  theme: {
    colors: {
      mojo: {
        primary: '#60A5FA',    // Blue
        secondary: '#A78BFA',  // Purple  
        accent: '#F472B6',     // Pink
        success: '#10B981',    // Green
        warning: '#F59E0B',    // Amber
        error: '#EF4444'       // Red
      }
    },
    fontFamily: {
      mono: ['JetBrains Mono', 'Fira Code', 'monospace']
    }
  },
  
  // Custom shortcuts for common patterns
  shortcuts: {
    // Performance badges
    'perf-badge-excellent': 'badge badge-success text-white font-mono',
    'perf-badge-good': 'badge badge-warning text-white font-mono',
    'perf-badge-poor': 'badge badge-error text-white font-mono',
    
    // Search interface
    'search-input': 'input input-bordered input-lg w-full focus:ring-2 focus:ring-primary focus:border-primary',
    'search-button': 'btn btn-primary btn-lg hover:scale-105 transform transition-transform',
    
    // Result cards
    'result-card': 'card bg-base-100 shadow-lg hover:shadow-xl transition-all duration-200 hover:scale-[1.02]',
    'result-header': 'flex justify-between items-start mb-2',
    'result-badges': 'flex gap-2 flex-wrap',
    'result-code': 'bg-base-200 p-4 rounded-lg overflow-x-auto text-sm font-mono',
    
    // Performance metrics
    'metric-card': 'card bg-base-100 border border-base-300',
    'metric-value': 'text-2xl font-bold font-mono',
    'metric-label': 'text-sm opacity-70',
    
    // Gradients
    'gradient-text': 'bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent',
    'gradient-bg': 'bg-gradient-to-br from-base-100 to-base-200',
    
    // Animations
    'pulse-slow': 'animate-pulse duration-2000',
    'glow': 'shadow-lg shadow-primary/50',
    'glow-success': 'shadow-lg shadow-success/50',
    'glow-warning': 'shadow-lg shadow-warning/50',
    
    // Layout
    'container-main': 'max-w-7xl mx-auto px-4 py-8',
    'section-spacing': 'space-y-8',
    'grid-responsive': 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
  },
  
  // Custom rules for specific use cases
  rules: [
    // Performance indicators
    [/^perf-(\d+)$/, ([, d]) => ({
      '--perf-value': d,
      'background': d > 90 ? '#10B981' : d > 70 ? '#F59E0B' : '#EF4444'
    })],
    
    // Dynamic latency colors
    [/^latency-(\d+)$/, ([, ms]) => ({
      'color': ms < 10 ? '#10B981' : ms < 50 ? '#F59E0B' : '#EF4444'
    })],
    
    // Similarity score backgrounds
    [/^similarity-(\d+)$/, ([, score]) => ({
      'background': `hsl(${score * 120 / 100}, 70%, 50%)`
    })]
  ],
  
  // Safelist for dynamic classes
  safelist: [
    'badge-success',
    'badge-warning', 
    'badge-error',
    'badge-info',
    'badge-primary',
    'badge-secondary',
    'text-success',
    'text-warning',
    'text-error',
    'bg-success',
    'bg-warning',
    'bg-error',
    'loading-spinner',
    'loading-dots',
    'loading-ring'
  ],
  
  // Transformers for better DX
  transformers: [
    // Add any custom transformers here
  ],
  
  // Content detection for auto-completion
  content: {
    filesystem: [
      'components/**/*.{vue,ts,tsx}',
      'pages/**/*.{vue,ts,tsx}',
      'layouts/**/*.{vue,ts,tsx}',
      'app.vue',
      'index.html'
    ]
  }
})

// Export utility functions for JavaScript usage
export const searchTheme = {
  colors: {
    primary: '#60A5FA',
    secondary: '#A78BFA', 
    accent: '#F472B6',
    success: '#10B981',
    warning: '#F59E0B',
    error: '#EF4444'
  },
  
  // Performance thresholds
  latencyThresholds: {
    excellent: 10,  // < 10ms
    good: 50,       // < 50ms  
    poor: 100       // > 100ms
  },
  
  // Similarity score colors
  similarityColors: (score: number) => {
    if (score > 0.9) return 'badge-success'
    if (score > 0.7) return 'badge-warning'
    return 'badge-error'
  },
  
  // Language specific colors
  languageColors: {
    typescript: 'badge-primary',
    javascript: 'badge-warning', 
    python: 'badge-success',
    go: 'badge-info',
    mojo: 'badge-secondary',
    rust: 'badge-error'
  }
}