#!/bin/bash

echo "🤖 DocChat RAG - Quick Start"
echo "=============================="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✓ Python found"

# Setup
echo
echo "1) Full setup (install + add docs + run)"
echo "2) Just install dependencies"
echo "3) Run API"
echo "4) Interactive CLI"
read -p "Choose [1-4]: " choice

case $choice in
    1)
        echo
        # Create venv
        python3 -m venv venv
        source venv/bin/activate
        
        # Install
        pip install -q -r requirements.txt
        echo "✓ Dependencies installed"
        
        # Setup .env
        if [ ! -f .env ]; then
            cp .env.example .env
            echo "⚠ Edit .env and add your OPENAI_API_KEY"
            read -p "Press Enter after setting API key..."
        fi
        
        # Add documents
        if [ $# -gt 0 ]; then
            echo "Adding documents..."
            python rag_engine.py "$@"
        fi
        
        # Run API
        echo "Starting API..."
        python app.py
        ;;
        
    2)
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        echo "✓ Setup complete!"
        echo "Run: source venv/bin/activate"
        ;;
        
    3)
        if [ ! -f .env ]; then
            echo "❌ .env not found. Run setup first."
            exit 1
        fi
        source venv/bin/activate 2>/dev/null || true
        python app.py
        ;;
        
    4)
        if [ $# -eq 0 ]; then
            echo "Usage: ./quickstart.sh 4 document.pdf"
            exit 1
        fi
        source venv/bin/activate 2>/dev/null || true
        python rag_engine.py "$@"
        ;;
        
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
