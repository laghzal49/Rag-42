#!/bin/bash
# ~/Rag-42/setup.sh - Smart project setup

set -e

echo "🔧 Setting up RAG project..."

# Check if we're in the right place
if [ ! -f "pyproject.toml" ]; then
  echo "❌ Run this from the project root"
  exit 1
fi

# Use RAG_BASE from environment (set by ~/.42rc)
if [ -z "$RAG_BASE" ]; then
  echo "⚠️  RAG_BASE not set, using local directory"
  RAG_BASE="."
fi

# Create symlinks to the big directories
echo "📦 Creating symlinks to $RAG_BASE..."

# Data directory
mkdir -p "$RAG_BASE/data"
ln -sf "$RAG_BASE/data" data

# Virtual environment (in RAG_BASE)
mkdir -p "$RAG_BASE/venv"
ln -sf "$RAG_BASE/venv" .venv

# Cache (already set by environment)
ln -sf "$RAG_BASE/cache" cache

# Set environment for this session
export HF_HOME="$RAG_BASE/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

# Create virtual environment
echo "🐍 Creating virtual environment..."
uv venv --python 3.10 .venv

# Activate
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate: source ~/Rag-42/.venv/bin/activate"
echo "To run: uv run python -m src <command>"
echo ""
echo "📊 Directory structure:"
echo "  Code: ~/Rag-42/ (in home)"
echo "  Data: $RAG_BASE/data (symlinked)"
echo "  Venv: $RAG_BASE/venv (symlinked)"
echo "  Cache: $RAG_BASE/cache (symlinked)"
