# Data Files with Local Paths

⚠️ **Note**: The following data files contain hardcoded local paths as part of their content:

- `analysis/vector_db_analysis.json` - Contains analysis results with absolute file paths
- `data/real_vector_corpus.json` - Contains extracted code snippets with source paths  
- `data/portfolio_corpus.json` - Contains portfolio analysis with project paths

These paths are **part of the data content** (file paths in code snippets, analysis results) and should not be modified as they represent the actual extraction source locations.

## For Cross-Project Usage

When using this package in other projects:

1. **Generate new corpus data** using your own project paths
2. **Use the configurable APIs** that accept custom corpus paths:
   ```python
   from src.integration import MCPOptimizedBridge
   
   bridge = MCPOptimizedBridge(
       corpus_path="/your/project/data/corpus.json",
       project_root="/your/project"
   )
   ```

3. **Consider the data files as examples** rather than production data

## Cleaning Up for Distribution

If preparing this repository for public distribution, consider:
- Moving data files to `.gitignore`
- Providing sample/anonymized data instead
- Creating data generation scripts that work with any project structure