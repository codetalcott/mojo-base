# Elysiajs Integration Guide

## 🔗 Integrating Mojo Semantic Search with Your Framework

This guide shows how to integrate the semantic search interface into your existing **Elysiajs + DaisyUI + UnoCSS** framework.

## 📂 Framework Integration

### 1. Copy Components to Your Project

```bash
# Copy the React component
cp web/components/SearchInterface.tsx your-project/src/components/

# Copy UnoCSS configuration  
cp web/uno.config.ts your-project/

# Update your existing uno.config.ts with our shortcuts and theme
```

### 2. Elysiajs API Routes

Add these routes to your Elysiajs server:

```typescript
// routes/search.ts
import { Elysia } from 'elysia'

export const searchRoutes = new Elysia({ prefix: '/api/search' })
  .get('/health', async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      const data = await response.json()
      return { status: 'healthy', mojo_api: data }
    } catch (error) {
      return { status: 'error', error: error.message }
    }
  })
  
  .get('/simple', async ({ query }) => {
    try {
      const params = new URLSearchParams(query)
      const response = await fetch(`http://localhost:8000/search/simple?${params}`)
      return await response.json()
    } catch (error) {
      return { error: error.message, results: [] }
    }
  })
  
  .post('/advanced', async ({ body }) => {
    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      return await response.json()
    } catch (error) {
      return { error: error.message, results: [] }
    }
  })
  
  .get('/corpus/stats', async () => {
    try {
      const response = await fetch('http://localhost:8000/corpus/stats')
      return await response.json()
    } catch (error) {
      return { error: error.message }
    }
  })
  
  .get('/performance/validate', async () => {
    try {
      const response = await fetch('http://localhost:8000/performance/validate')
      return await response.json()
    } catch (error) {
      return { error: error.message }
    }
  })
```

### 3. Add to Your Main App

```typescript
// app.ts or main.ts
import { Elysia } from 'elysia'
import { searchRoutes } from './routes/search'

const app = new Elysia()
  .use(searchRoutes)
  // ... your existing routes
  
export default app
```

### 4. Update UnoCSS Config

Merge the semantic search styles into your existing UnoCSS configuration:

```typescript
// your-project/uno.config.ts
import { defineConfig } from 'unocss'
import { presetDaisy } from 'unocss-preset-daisy'
import { searchTheme } from './web/uno.config' // Import our theme

export default defineConfig({
  presets: [
    // your existing presets
    presetDaisy({
      themes: ['dark', 'light', 'cyberpunk'] // Add cyberpunk theme
    })
  ],
  
  theme: {
    extend: {
      colors: {
        // Merge with your existing colors
        ...searchTheme.colors
      }
    }
  },
  
  shortcuts: {
    // Add our search-specific shortcuts
    'search-input': 'input input-bordered input-lg w-full focus:ring-2 focus:ring-primary',
    'result-card': 'card bg-base-100 shadow-lg hover:shadow-xl transition-all duration-200',
    'gradient-text': 'bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent',
    // ... your existing shortcuts
  }
})
```

### 5. Use in Your Pages

```typescript
// pages/search.tsx or wherever you want the search
import { SearchInterface } from '../components/SearchInterface'

export default function SearchPage() {
  return (
    <div className="min-h-screen bg-base-300">
      <SearchInterface 
        apiBaseUrl="/api/search" // Use your Elysiajs routes
        className="container-main" // Use your layout classes
      />
    </div>
  )
}
```

## 🎨 Styling Integration

### DaisyUI Theme Integration

The interface uses these DaisyUI components that should work with your existing theme:

- `card`, `card-body`, `card-title`
- `input`, `input-bordered`, `input-lg`
- `btn`, `btn-primary`, `btn-ghost`
- `badge`, `badge-success`, `badge-outline`
- `stats`, `stat`, `stat-title`, `stat-value`
- `loading`, `loading-spinner`
- `progress`, `progress-primary`

### Custom CSS Variables

Add these to your global CSS for consistent theming:

```css
:root {
  --mojo-primary: #60A5FA;
  --mojo-secondary: #A78BFA;
  --mojo-accent: #F472B6;
  --mojo-success: #10B981;
  --search-glow: 0 0 20px rgba(96, 165, 250, 0.5);
}

.search-glow {
  box-shadow: var(--search-glow);
}

.gradient-text {
  background: linear-gradient(to right, var(--mojo-primary), var(--mojo-secondary), var(--mojo-accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
```

## 🚀 Deployment

### Development

```bash
# Start your Elysiajs server
bun run dev

# Start the Mojo API server (in separate terminal)
python3 api/semantic_search_api_v2.py

# Your search will be available at your normal dev URL + /search
```

### Production

```bash
# Build your project with the integrated search
bun run build

# Deploy both:
# 1. Your Elysiajs app (with search routes)
# 2. Mojo API server (on different port/service)

# Update apiBaseUrl in SearchInterface to point to production Mojo API
```

## 🔧 Configuration

### Environment Variables

Add these to your `.env`:

```bash
MOJO_API_URL=http://localhost:8000  # Development
# MOJO_API_URL=https://your-mojo-api.com  # Production

SEARCH_CACHE_TTL=300  # 5 minutes
SEARCH_MAX_RESULTS=20
SEARCH_ENABLE_MCP=true
```

### Feature Flags

```typescript
// config/search.ts
export const searchConfig = {
  enableMCP: process.env.SEARCH_ENABLE_MCP === 'true',
  maxResults: parseInt(process.env.SEARCH_MAX_RESULTS || '10'),
  cacheTTL: parseInt(process.env.SEARCH_CACHE_TTL || '300'),
  apiUrl: process.env.MOJO_API_URL || 'http://localhost:8000'
}
```

## 📱 Mobile Responsiveness

The interface is built with mobile-first design using DaisyUI responsive classes:

- `grid-cols-1 md:grid-cols-2 lg:grid-cols-3` for responsive grids
- `flex-col md:flex-row` for responsive layouts
- `text-sm md:text-base lg:text-lg` for responsive typography

## 🎯 Performance Optimization

### For Your Framework

1. **Code Splitting**: Lazy load the search component
2. **Caching**: Cache search results in your state management
3. **Debouncing**: Add search debouncing for better UX
4. **SSR**: Pre-render the search interface for SEO

```typescript
// Lazy loading example
const SearchInterface = lazy(() => import('../components/SearchInterface'))

// In your component
<Suspense fallback={<div className="loading loading-spinner"></div>}>
  <SearchInterface />
</Suspense>
```

## 🧪 Testing

Test the integration:

1. **API Routes**: Test all `/api/search/*` endpoints
2. **Component**: Test search functionality
3. **Styling**: Verify DaisyUI theme consistency
4. **Performance**: Test search latency through your routes

```typescript
// Example test
import { describe, it, expect } from 'bun:test'
import app from '../app'

describe('Search API', () => {
  it('should proxy search requests', async () => {
    const response = await app.handle(
      new Request('http://localhost/api/search/simple?q=test')
    )
    expect(response.status).toBe(200)
  })
})
```

## 🎉 Ready for Integration!

This setup gives you:
- ✅ Seamless integration with your existing framework
- ✅ Consistent DaisyUI styling
- ✅ UnoCSS optimization  
- ✅ Production-ready configuration
- ✅ Mobile responsive design
- ✅ Performance optimized

The semantic search will feel like a native part of your application! 🚀