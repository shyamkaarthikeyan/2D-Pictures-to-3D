"""TripoSR Web Application"""

from flask import Flask
from flask_cors import CORS
from app.core.model_loader import initialize_model
from app.core.config import Config
import logging

def create_app():
    """Create Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the model
    initialize_model()
    
    # Register all blueprints
    from app.api.routes import api_bp
    from app.api.qr_routes import qr_bp
    from app.main import main_bp
    
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(qr_bp, url_prefix='/api')
    app.register_blueprint(main_bp)
    
    return app
