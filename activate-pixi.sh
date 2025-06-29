#!/bin/bash
# Activate pixi environment for Mojo development

echo "🚀 Activating Pixi Environment for Mojo Development"
echo "=================================================="

cd <project-root>/portfolio-search

# Set environment variables from pixi shell-hook
export PATH="<project-root>/portfolio-search/.pixi/envs/default/bin:/Users/williamtalcott/.pixi/bin:$PATH"
export CONDA_PREFIX="<project-root>/portfolio-search/.pixi/envs/default"
export PIXI_PROJECT_NAME="portfolio-search"
export PIXI_IN_SHELL="1"
export PIXI_EXE="/Users/williamtalcott/.pixi/bin/pixi"
export PIXI_PROJECT_ROOT="<project-root>/portfolio-search"
export PIXI_PROJECT_VERSION="0.1.0"
export PIXI_PROJECT_MANIFEST="<project-root>/portfolio-search/pixi.toml"
export CONDA_DEFAULT_ENV="portfolio-search"
export PIXI_ENVIRONMENT_NAME="default"
export PIXI_ENVIRONMENT_PLATFORMS="osx-arm64"
export PIXI_PROMPT="(portfolio-search) "

# Source activation scripts
if [ -f "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/10-activate-max.sh" ]; then
    source "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/10-activate-max.sh"
fi

if [ -f "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/libarrow_activate.sh" ]; then
    source "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/libarrow_activate.sh"
fi

if [ -f "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/libxml2_activate.sh" ]; then
    source "<project-root>/portfolio-search/.pixi/envs/default/etc/conda/activate.d/libxml2_activate.sh"
fi

echo "✅ Pixi environment activated!"
echo "📊 Environment details:"
echo "  - Project: $PIXI_PROJECT_NAME"
echo "  - Version: $PIXI_PROJECT_VERSION" 
echo "  - Platform: $PIXI_ENVIRONMENT_PLATFORMS"
echo "  - Conda Prefix: $CONDA_PREFIX"

echo ""
echo "🔧 Available commands:"
echo "  - mojo --version     # Check Mojo installation"
echo "  - max --version      # Check MAX installation"
echo "  - pixi run <command> # Run commands in environment"

echo ""
echo "🎯 Ready for Mojo development!"