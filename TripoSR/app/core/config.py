"""Configuration settings"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'triposr-secret-key'
    
    # Model settings
    MODEL_NAME = "stabilityai/TripoSR"
    CONFIG_NAME = "config.yaml"
    WEIGHT_NAME = "model.ckpt"
    CHUNK_SIZE = 8192
    
    # Processing settings
    MC_RESOLUTION = 256
    FOREGROUND_RATIO = 0.85
    
    # File settings
    OUTPUT_DIR = Path("outputs")
    
    # Session settings (for QR code)
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_IMAGES_PER_SESSION = 4
    
    @classmethod
    def init_directories(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
