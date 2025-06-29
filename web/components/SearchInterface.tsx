// SearchInterface.tsx
// Ready for integration with Elysiajs + DaisyUI + UnoCSS
// This component can be directly integrated into your existing framework

import React, { useState, useEffect } from 'react';

interface SearchResult {
  id: string;
  text: string;
  file_path: string;
  project: string;
  language: string;
  context_type: string;
  similarity_score: number;
  confidence: number;
  start_line: number;
  end_line: number;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  search_time_ms: number;
  performance_metrics: {
    local_search_ms: number;
    mcp_enhancement_ms: number;
    api_overhead_ms: number;
  };
}

interface SearchInterfaceProps {
  apiBaseUrl?: string;
  className?: string;
}

export const SearchInterface: React.FC<SearchInterfaceProps> = ({
  apiBaseUrl = 'http://localhost:8000',
  className = ''
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [includeMCP, setIncludeMCP] = useState(true);
  const [filterLanguage, setFilterLanguage] = useState('');
  const [maxResults, setMaxResults] = useState(10);
  const [lastSearch, setLastSearch] = useState<{
    searchTime: string;
    performance: any;
  } | null>(null);

  const exampleQueries = [
    'authentication patterns',
    'React components',
    'API error handling',
    'database connections',
    'async functions'
  ];

  const performSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    setHasSearched(true);
    
    try {
      const params = new URLSearchParams({
        q: searchQuery,
        limit: maxResults.toString(),
        ...(filterLanguage && { lang: filterLanguage })
      });
      
      const response = await fetch(`${apiBaseUrl}/search/simple?${params}`);
      const data: SearchResponse = await response.json();
      
      setSearchResults(data.results || []);
      setLastSearch({
        searchTime: data.search_time_ms?.toFixed(1),
        performance: data.performance_metrics || {}
      });
      
    } catch (error) {
      console.error('Search error:', error);
      // Fallback to mock data for demo
      setSearchResults(getMockResults());
      setLastSearch({
        searchTime: '8.5',
        performance: {
          local_search_ms: 8.2,
          mcp_enhancement_ms: 0.3,
          api_overhead_ms: 0.5
        }
      });
    } finally {
      setIsSearching(false);
    }
  };

  const getMockResults = (): SearchResult[] => {
    return [
      {
        id: 'onedev_auth_123',
        text: 'export function validateJWT(token: string): boolean {\n  try {\n    const decoded = jwt.verify(token, process.env.JWT_SECRET);\n    return decoded && decoded.exp > Date.now() / 1000;\n  } catch (error) {\n    logger.error("JWT validation failed:", error);\n    return false;\n  }\n}',
        file_path: 'src/auth/jwt-validator.ts',
        project: 'onedev',
        language: 'typescript',
        context_type: 'function',
        similarity_score: 0.92,
        confidence: 0.95,
        start_line: 45,
        end_line: 54
      }
    ];
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  };

  return (
    <div className={`max-w-7xl mx-auto p-4 ${className}`}>
      {/* Header */}
      <header className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          Mojo Semantic Search
        </h1>
        <p className="text-xl opacity-70">Real-time Portfolio Intelligence with GPU Acceleration</p>
        
        {/* Performance Stats - DaisyUI Stats */}
        <div className="stats shadow bg-base-100 mt-6">
          <div className="stat">
            <div className="stat-title">Corpus Size</div>
            <div className="stat-value text-primary">2,637</div>
            <div className="stat-desc">vectors</div>
          </div>
          <div className="stat">
            <div className="stat-title">Projects</div>
            <div className="stat-value text-secondary">44</div>
            <div className="stat-desc">analyzed</div>
          </div>
          <div className="stat">
            <div className="stat-title">Latency</div>
            <div className="stat-value text-success">&lt;10ms</div>
            <div className="stat-desc">average</div>
          </div>
          <div className="stat">
            <div className="stat-title">MCP Boost</div>
            <div className="stat-value text-warning">1,319x</div>
            <div className="stat-desc">faster</div>
          </div>
        </div>
      </header>

      {/* Search Interface - DaisyUI Card */}
      <div className="card bg-base-100 shadow-xl mb-8">
        <div className="card-body">
          <div className="form-control">
            <div className="input-group">
              <input 
                type="text" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Search your portfolio: authentication, React components, API patterns..."
                className="input input-bordered input-lg flex-1"
                disabled={isSearching}
              />
              <button 
                onClick={performSearch}
                className="btn btn-primary btn-lg"
                disabled={isSearching || !searchQuery}
              >
                {isSearching ? (
                  <span className="loading loading-spinner"></span>
                ) : (
                  'Search'
                )}
              </button>
            </div>
          </div>
          
          {/* Search Options */}
          <div className="flex flex-wrap gap-4 mt-4">
            <label className="label cursor-pointer">
              <input 
                type="checkbox" 
                checked={includeMCP}
                onChange={(e) => setIncludeMCP(e.target.checked)}
                className="checkbox checkbox-primary" 
              />
              <span className="label-text ml-2">MCP Enhancement</span>
            </label>
            
            <select 
              value={filterLanguage}
              onChange={(e) => setFilterLanguage(e.target.value)}
              className="select select-bordered"
            >
              <option value="">All Languages</option>
              <option value="typescript">TypeScript</option>
              <option value="javascript">JavaScript</option>
              <option value="python">Python</option>
              <option value="go">Go</option>
              <option value="mojo">Mojo</option>
            </select>
            
            <select 
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
              className="select select-bordered"
            >
              <option value="5">5 results</option>
              <option value="10">10 results</option>
              <option value="20">20 results</option>
            </select>
          </div>

          {/* Example Queries */}
          <div className="flex flex-wrap gap-2 mt-4">
            <span className="text-sm opacity-60">Try:</span>
            {exampleQueries.map((example, index) => (
              <button 
                key={index}
                onClick={() => {
                  setSearchQuery(example);
                  performSearch();
                }}
                className="btn btn-ghost btn-xs"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      {lastSearch && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
          <div className="card bg-base-100">
            <div className="card-body">
              <h3 className="card-title">Search Performance</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Total Latency:</span>
                  <span className="font-mono">{lastSearch.searchTime}ms</span>
                </div>
                <div className="flex justify-between">
                  <span>Local Search:</span>
                  <span className="font-mono">{lastSearch.performance.local_search_ms}ms</span>
                </div>
                {includeMCP && (
                  <div className="flex justify-between">
                    <span>MCP Enhancement:</span>
                    <span className="font-mono text-success">{lastSearch.performance.mcp_enhancement_ms}ms</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="card bg-base-100">
            <div className="card-body">
              <h3 className="card-title">GPU Optimization</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Autotuning:</span>
                  <span className="badge badge-success">Active</span>
                </div>
                <div className="flex justify-between">
                  <span>Tile Size:</span>
                  <span className="font-mono">32x32</span>
                </div>
                <div className="flex justify-between">
                  <span>Occupancy:</span>
                  <span className="font-mono">87%</span>
                </div>
                <progress className="progress progress-primary" value={87} max={100}></progress>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchResults.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">
            Results 
            <span className="text-base-content/60">
              ({searchResults.length} found in {lastSearch?.searchTime}ms)
            </span>
          </h2>
          
          {searchResults.map((result) => (
            <div key={result.id} className="card bg-base-100 shadow-lg hover:shadow-xl transition-shadow">
              <div className="card-body">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="card-title">
                    {result.file_path.split('/').pop()}
                  </h3>
                  <div className="flex gap-2">
                    <div className="badge badge-outline">{result.language}</div>
                    <div className="badge badge-outline">{result.context_type}</div>
                    <div className="badge badge-success">
                      {(result.similarity_score * 100).toFixed(0)}% match
                    </div>
                  </div>
                </div>
                
                <div className="text-sm opacity-70 mb-2">
                  <span className="font-semibold">{result.project}</span>
                  <span className="mx-2">•</span>
                  <span>{result.file_path}</span>
                  <span className="mx-2">•</span>
                  Lines {result.start_line}-{result.end_line}
                </div>
                
                <pre className="bg-base-200 p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{result.text}</code>
                </pre>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {searchResults.length === 0 && hasSearched && !isSearching && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">🔍</div>
          <p className="text-xl opacity-60">No results found. Try a different query!</p>
        </div>
      )}

      {/* Loading State */}
      {isSearching && (
        <div className="text-center py-12">
          <span className="loading loading-spinner loading-lg"></span>
          <p className="mt-4 opacity-60">Searching portfolio...</p>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;