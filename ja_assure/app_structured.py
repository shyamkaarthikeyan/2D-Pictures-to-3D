#!/usr/bin/env python3
"""
TripoSR Web Application - Restructured
Main entry point for the Flask application
"""

import argparse
import logging
from app import create_app

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='TripoSR Web Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create Flask application
    app = create_app()
    
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting TripoSR Web Application on {args.host}:{args.port}")
    
    # Run the application
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
