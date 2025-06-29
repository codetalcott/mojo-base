"""
Integration bridge with onedev portfolio intelligence system.
Leverages onedev's MCP tools for enhanced semantic search capabilities.

This module gracefully degrades to basic functionality when onedev is not available.
"""

from utils.list import List
from ..core.data_structures import CodeSnippet, SearchContext
from ..search.semantic_search_engine import SemanticSearchEngine

struct OnedevBridge:
    """
    Optional bridge to onedev portfolio intelligence system.
    
    Capabilities when onedev available:
    - Portfolio project scanning and indexing
    - Context assembly from multiple projects
    - Architectural pattern detection
    - Cross-project dependency analysis
    
    Fallback capabilities when onedev unavailable:
    - Basic project scanning from local filesystem
    - Simple context assembly
    - Local pattern detection
    """
    var search_engine: SemanticSearchEngine
    var portfolio_projects: List[String]
    var onedev_mcp_path: String
    var integration_available: Bool
    var fallback_mode: Bool
    var config_path: String
    
    fn __init__(inout self, search_engine: SemanticSearchEngine, config_path: String = "config.json"):
        """Initialize onedev integration bridge with optional fallback."""
        self.search_engine = search_engine
        self.portfolio_projects = List[String]()
        self.onedev_mcp_path = "<onedev-project-path>/dist/infrastructure/mcp/unified-mcp-main-v2.js"
        self.integration_available = False
        self.fallback_mode = True
        self.config_path = config_path
        
        # Load configuration and test onedev availability
        self._load_config()
        self._detect_onedev_availability()
    
    fn _load_config(inout self):
        """Load configuration from config.json."""
        # TODO: Implement JSON config loading
        # For now, default to safe fallback mode
        print("🔧 Loading configuration...")
        print("📄 Config: Onedev integration set to fallback mode")
    
    fn _detect_onedev_availability(inout self) -> Bool:
        """Auto-detect if onedev is available and working."""
        print("🔍 Detecting onedev availability...")
        
        # TODO: Implement actual onedev detection
        # Check if onedev MCP server is running
        # Check if onedev project exists at expected path
        # Test basic MCP communication
        
        self.integration_available = False  # Default to false for public usage
        
        if self.integration_available:
            print("✅ Onedev integration: AVAILABLE")
            self.fallback_mode = False
        else:
            print("⚠️  Onedev integration: NOT AVAILABLE - using fallback mode")
            print("   This provides basic functionality without onedev dependency")
            self.fallback_mode = True
        
        return self.integration_available
    
    fn scan_portfolio_projects(inout self) -> Int:
        """
        Scan portfolio for projects using onedev tools or fallback method.
        
        Returns:
            Number of projects discovered and indexed
        """
        if self.integration_available:
            return self._scan_with_onedev()
        else:
            return self._scan_with_fallback()
    
    fn _scan_with_onedev(inout self) -> Int:
        """Scan using onedev MCP tools."""
        print("🔍 Scanning portfolio with onedev integration...")
        
        # Use onedev's scan_projects MCP tool
        let discovered_projects = self._call_onedev_scan_projects()
        
        var indexed_count = 0
        for project in discovered_projects:
            if self._index_project(project):
                indexed_count += 1
                self.portfolio_projects.append(project.name)
        
        print("📊 Onedev scan completed:", indexed_count, "projects indexed")
        return indexed_count
    
    fn _scan_with_fallback(inout self) -> Int:
        """Scan using basic filesystem methods."""
        print("🔍 Scanning portfolio with fallback method...")
        print("   (Limited functionality without onedev)")
        
        # Basic project discovery from common locations
        let common_project_paths = [
            "./",
            "../",
            "../../projects/",
            "~/projects/"
        ]
        
        var indexed_count = 0
        
        # Look for common project indicators (package.json, cargo.toml, etc.)
        for base_path in common_project_paths:
            let projects = self._discover_local_projects(base_path)
            for project in projects:
                if self._index_project_basic(project):
                    indexed_count += 1
                    self.portfolio_projects.append(project)
        
        print("📊 Fallback scan completed:", indexed_count, "projects indexed")
        return indexed_count
    
    fn _discover_local_projects(self, base_path: String) -> List[String]:
        """Discover projects in local filesystem."""
        var projects = List[String]()
        
        # TODO: Implement filesystem scanning
        # Look for package.json, Cargo.toml, pyproject.toml, etc.
        # For now, return empty list
        
        return projects
    
    fn _index_project_basic(inout self, project_path: String) -> Bool:
        """Basic project indexing without onedev."""
        # TODO: Implement basic file parsing and indexing
        # This would scan common file types and extract functions/classes
        return False
    
    fn assemble_search_context(inout self, 
                              query: String,
                              focus: String = "general") -> SearchContext:
        """
        Assemble search context using onedev or fallback method.
        
        Args:
            query: Search query for context
            focus: Search focus ("api", "patterns", "implementations")
            
        Returns:
            Enhanced search context (full with onedev, basic without)
        """
        var context = SearchContext()
        context.set_focus(focus)
        
        if self.integration_available:
            return self._assemble_context_with_onedev(query, focus, context)
        else:
            return self._assemble_context_fallback(query, focus, context)
    
    fn _assemble_context_with_onedev(inout self, 
                                   query: String, 
                                   focus: String, 
                                   inout context: SearchContext) -> SearchContext:
        """Assemble context with full onedev integration."""
        print("🧠 Assembling context with onedev intelligence...")
        
        # Use onedev's assemble_context MCP tool
        let onedev_context = self._call_onedev_assemble_context(query, focus)
        
        # Enhance context with onedev intelligence
        context.current_project = onedev_context.suggested_project
        context.preferred_languages = onedev_context.relevant_languages
        
        # Add architectural insights
        for pattern in onedev_context.architectural_patterns:
            context.add_recent_query(pattern)
        
        return context
    
    fn _assemble_context_fallback(inout self,
                                query: String,
                                focus: String, 
                                inout context: SearchContext) -> SearchContext:
        """Assemble basic context without onedev."""
        print("🧠 Assembling basic context (fallback mode)...")
        
        # Basic context assembly based on query analysis
        let query_lower = query.lower()
        
        # Detect likely programming languages from query
        if "typescript" in query_lower or "ts" in query_lower:
            context.preferred_languages.append("typescript")
        if "python" in query_lower or "py" in query_lower:
            context.preferred_languages.append("python")
        if "javascript" in query_lower or "js" in query_lower:
            context.preferred_languages.append("javascript")
        if "mojo" in query_lower:
            context.preferred_languages.append("mojo")
        
        # Basic focus-based context
        if focus == "api":
            context.add_recent_query("API patterns")
        elif focus == "patterns":
            context.add_recent_query("Design patterns")
        elif focus == "implementations":
            context.add_recent_query("Implementation examples")
        
        return context
    
    fn find_architectural_patterns(inout self, pattern_type: String) -> List[CodeSnippet]:
        """
        Find architectural patterns using onedev intelligence.
        
        Args:
            pattern_type: "middleware", "database", "api", "auth", etc.
            
        Returns:
            Code snippets matching architectural patterns across projects
        """
        var patterns = List[CodeSnippet]()
        
        if self.integration_active:
            # Use onedev's find_patterns MCP tool
            let onedev_patterns = self._call_onedev_find_patterns(pattern_type)
            
            for pattern_data in onedev_patterns:
                let snippet = CodeSnippet(
                    content=pattern_data.code,
                    file_path=pattern_data.file_path,
                    project_name=pattern_data.project_name,
                    function_name=pattern_data.function_name,
                    line_start=pattern_data.line_start,
                    line_end=pattern_data.line_end
                )
                
                # Add dependencies from onedev analysis
                for dep in pattern_data.dependencies:
                    snippet.add_dependency(dep)
                
                patterns.append(snippet)
        
        return patterns
    
    fn get_vector_similarity_insights(inout self, 
                                    query: String) -> OnedevVectorInsights:
        """
        Get vector similarity insights from onedev.
        
        Args:
            query: Query for similarity analysis
            
        Returns:
            Vector-based insights and recommendations
        """
        var insights = OnedevVectorInsights()
        
        if self.integration_active:
            # Use onedev's get_vector_similarity_insights MCP tool
            insights = self._call_onedev_vector_insights(query)
        
        return insights
    
    fn get_architectural_recommendations(inout self,
                                       context: String) -> List[String]:
        """
        Get architectural recommendations from onedev.
        
        Args:
            context: Current development context
            
        Returns:
            List of architectural recommendations
        """
        var recommendations = List[String]()
        
        if self.integration_active:
            # Use onedev's get_architectural_recommendations MCP tool
            recommendations = self._call_onedev_arch_recommendations(context)
        
        return recommendations
    
    fn enhance_search_with_portfolio_intelligence(inout self,
                                                 query: String,
                                                 base_results: List[SearchResult]) -> List[SearchResult]:
        """
        Enhance search results with onedev portfolio intelligence.
        
        Args:
            query: Original search query
            base_results: Results from semantic search
            
        Returns:
            Enhanced results with portfolio context
        """
        var enhanced_results = base_results
        
        if self.integration_active:
            # Get portfolio insights for query
            let portfolio_insights = self._call_onedev_portfolio_analysis(query)
            
            # Apply portfolio-based ranking adjustments
            for result in enhanced_results:
                # Boost results from high-health projects
                if result.snippet.project_name in portfolio_insights.high_health_projects:
                    result.project_relevance *= 1.2
                
                # Boost results from actively developed projects
                if result.snippet.project_name in portfolio_insights.active_projects:
                    result.recency_boost *= 1.1
                
                # Apply technology relevance boost
                for tech in portfolio_insights.relevant_technologies:
                    if tech in result.snippet.dependencies:
                        result.context_relevance *= 1.1
                
                # Recalculate final score
                result.calculate_final_score()
        
        return enhanced_results
    
    fn _index_project(inout self, project: ProjectInfo) -> Bool:
        """Index a single project's code snippets."""
        var success_count = 0
        
        # Process each file in the project
        for file_info in project.files:
            # Extract code snippets from file
            let snippets = self._extract_code_snippets(file_info, project.name)
            
            for snippet in snippets:
                if self.search_engine.index_code_snippet(snippet):
                    success_count += 1
        
        return success_count > 0
    
    fn _extract_code_snippets(self, file_info: FileInfo, project_name: String) -> List[CodeSnippet]:
        """Extract meaningful code snippets from a file."""
        var snippets = List[CodeSnippet]()
        
        # TODO: Implement AST-based extraction using Tree-sitter
        # For now, extract function-level snippets
        
        let functions = self._parse_functions(file_info.content)
        
        for func in functions:
            let snippet = CodeSnippet(
                content=func.body,
                file_path=file_info.path,
                project_name=project_name,
                function_name=func.name,
                line_start=func.line_start,
                line_end=func.line_end
            )
            snippets.append(snippet)
        
        return snippets
    
    fn _parse_functions(self, file_content: String) -> List[FunctionInfo]:
        """Parse functions from file content (simplified)."""
        var functions = List[FunctionInfo]()
        
        # TODO: Implement proper AST parsing
        # Simplified function detection for now
        
        return functions
    
    # Onedev MCP tool wrappers (placeholder implementations)
    fn _call_onedev_scan_projects(self) -> List[ProjectInfo]:
        """Call onedev scan_projects MCP tool."""
        # TODO: Implement actual MCP communication
        return List[ProjectInfo]()
    
    fn _call_onedev_assemble_context(self, query: String, focus: String) -> OnedevContext:
        """Call onedev assemble_context MCP tool."""
        # TODO: Implement actual MCP communication
        return OnedevContext()
    
    fn _call_onedev_find_patterns(self, pattern_type: String) -> List[PatternData]:
        """Call onedev find_patterns MCP tool."""
        # TODO: Implement actual MCP communication
        return List[PatternData]()
    
    fn _call_onedev_vector_insights(self, query: String) -> OnedevVectorInsights:
        """Call onedev vector similarity insights MCP tool."""
        # TODO: Implement actual MCP communication
        return OnedevVectorInsights()
    
    fn _call_onedev_arch_recommendations(self, context: String) -> List[String]:
        """Call onedev architectural recommendations MCP tool."""
        # TODO: Implement actual MCP communication
        return List[String]()
    
    fn _call_onedev_portfolio_analysis(self, query: String) -> PortfolioInsights:
        """Call onedev portfolio analysis MCP tool."""
        # TODO: Implement actual MCP communication
        return PortfolioInsights()
    
    fn get_integration_status(self) -> String:
        """Get integration status report."""
        if self.integration_available:
            return (
                "Onedev Integration Status:\n" +
                "- Mode: FULL INTEGRATION ✅\n" +
                "- Features: Portfolio scanning, context assembly, pattern detection\n" +
                "- Portfolio Projects: " + String(len(self.portfolio_projects)) + "\n" +
                "- Config Path: " + self.config_path + "\n" +
                "- MCP Path: " + self.onedev_mcp_path
            )
        else:
            return (
                "Onedev Integration Status:\n" +
                "- Mode: FALLBACK MODE ⚠️\n" +
                "- Features: Basic search, limited context assembly\n" +
                "- Note: Install onedev for full functionality\n" +
                "- Portfolio Projects: " + String(len(self.portfolio_projects)) + "\n" +
                "- Config Path: " + self.config_path
            )
    
    fn enable_onedev_if_available(inout self) -> Bool:
        """Try to enable onedev integration if it becomes available."""
        print("🔄 Re-checking onedev availability...")
        let was_available = self.integration_available
        self._detect_onedev_availability()
        
        if self.integration_available and not was_available:
            print("✅ Onedev integration now enabled!")
            return True
        elif not self.integration_available:
            print("⚠️  Onedev still not available - continuing in fallback mode")
        
        return self.integration_available
    
    fn get_capabilities(self) -> List[String]:
        """Get list of current capabilities based on integration status."""
        var capabilities = List[String]()
        
        # Always available capabilities
        capabilities.append("Basic semantic search")
        capabilities.append("Local project scanning")
        capabilities.append("Simple context assembly")
        
        if self.integration_available:
            # Enhanced capabilities with onedev
            capabilities.append("Portfolio intelligence")
            capabilities.append("Cross-project pattern detection")
            capabilities.append("Architectural recommendations")
            capabilities.append("Vector similarity insights")
            capabilities.append("Project health-based ranking")
        else:
            # Fallback limitations
            capabilities.append("Limited to basic functionality")
            capabilities.append("No cross-project insights")
        
        return capabilities

# Supporting data structures for onedev integration
struct ProjectInfo:
    var name: String
    var path: String
    var health_score: Float32
    var files: List[FileInfo]
    var technologies: List[String]
    
    fn __init__(inout self, name: String, path: String):
        self.name = name
        self.path = path
        self.health_score = 0.0
        self.files = List[FileInfo]()
        self.technologies = List[String]()

struct FileInfo:
    var path: String
    var content: String
    var language: String
    var last_modified: Int
    
    fn __init__(inout self, path: String, content: String):
        self.path = path
        self.content = content
        self.language = ""
        self.last_modified = 0

struct FunctionInfo:
    var name: String
    var body: String
    var line_start: Int
    var line_end: Int
    var parameters: List[String]
    
    fn __init__(inout self, name: String, body: String):
        self.name = name
        self.body = body
        self.line_start = 0
        self.line_end = 0
        self.parameters = List[String]()

struct OnedevContext:
    var suggested_project: String
    var relevant_languages: List[String]
    var architectural_patterns: List[String]
    
    fn __init__(inout self):
        self.suggested_project = ""
        self.relevant_languages = List[String]()
        self.architectural_patterns = List[String]()

struct OnedevVectorInsights:
    var similar_patterns: List[String]
    var related_projects: List[String]
    var confidence_score: Float32
    
    fn __init__(inout self):
        self.similar_patterns = List[String]()
        self.related_projects = List[String]()
        self.confidence_score = 0.0

struct PatternData:
    var code: String
    var file_path: String
    var project_name: String
    var function_name: String
    var line_start: Int
    var line_end: Int
    var dependencies: List[String]
    
    fn __init__(inout self):
        self.code = ""
        self.file_path = ""
        self.project_name = ""
        self.function_name = ""
        self.line_start = 0
        self.line_end = 0
        self.dependencies = List[String]()

struct PortfolioInsights:
    var high_health_projects: List[String]
    var active_projects: List[String]
    var relevant_technologies: List[String]
    var consolidation_opportunities: List[String]
    
    fn __init__(inout self):
        self.high_health_projects = List[String]()
        self.active_projects = List[String]()
        self.relevant_technologies = List[String]()
        self.consolidation_opportunities = List[String]()

# High-level API
fn create_onedev_bridge(search_engine: SemanticSearchEngine) -> OnedevBridge:
    """Create onedev integration bridge."""
    return OnedevBridge(search_engine)