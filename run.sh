#!/bin/bash
# Quick start script for LangChain Demo

echo "🚀 Starting LangChain Use Cases Demo..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit
streamlit run app.py

