#!/bin/bash

echo "========================================"
echo "      TripoSR Web Application"
echo "========================================"
echo

echo "Installing web dependencies..."
pip install -r web_requirements.txt

echo
echo "Starting TripoSR Web Application..."
echo "Open your browser to: http://localhost:5000"
echo
echo "Press Ctrl+C to stop the server"
echo

python web_app.py --host 0.0.0.0 --port 5000
