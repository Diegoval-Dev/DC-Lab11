#!/bin/bash

# Traffic Analytics GT Dashboard Runner
echo "Starting Traffic Analytics GT Dashboard..."

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit dashboard
streamlit run dashboard.py --server.port 8501 --server.address localhost

echo "Dashboard started at http://localhost:8501"