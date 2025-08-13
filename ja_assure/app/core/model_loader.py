"""Model initialization"""

import logging
import torch
import rembg
from tsr.system import TSR
from app.core.config import Config

logger = logging.getLogger(__name__)

# Global instances
model = None
rembg_session = None
device = None

def initialize_model():
    """Initialize TripoSR model"""
    global model, rembg_session, device
    
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        logger.info("Loading TripoSR model...")
        model = TSR.from_pretrained(
            Config.MODEL_NAME,
            config_name=Config.CONFIG_NAME,
            weight_name=Config.WEIGHT_NAME,
        )
        model.renderer.set_chunk_size(Config.CHUNK_SIZE)
        model.to(device)
        logger.info("Model loaded successfully")
        
        rembg_session = rembg.new_session()
        Config.init_directories()
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

def get_model():
    return model

def get_rembg_session():
    return rembg_session

def get_device_name():
    return device
