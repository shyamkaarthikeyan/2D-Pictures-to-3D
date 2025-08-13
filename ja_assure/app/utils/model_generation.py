"""
3D model generation utilities
"""

import time
import logging
import torch
from pathlib import Path
from app.core.model_loader import get_model, get_device_name
from app.core.config import Config
from tsr.utils import to_gradio_3d_orientation

logger = logging.getLogger(__name__)

def generate_3d_model(image, mc_resolution=None, formats=None):
    """
    Generate 3D model from preprocessed image
    
    Args:
        image: Preprocessed PIL Image
        mc_resolution: Marching cubes resolution
        formats: List of output formats ['obj', 'glb']
        
    Returns:
        tuple: (output_files dict, scene_codes)
    """
    if mc_resolution is None:
        mc_resolution = Config.MC_RESOLUTION
    
    if formats is None:
        formats = ["obj", "glb"]
    
    try:
        logger.info("Generating 3D model...")
        
        model = get_model()
        device = get_device_name()
        
        # Generate scene codes
        with torch.no_grad():
            scene_codes = model([image], device=device)
        
        # Extract mesh
        mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
        mesh = to_gradio_3d_orientation(mesh)
        
        # Save in requested formats
        output_files = {}
        timestamp = str(int(time.time()))
        
        for format_type in formats:
            filename = f"model_{timestamp}.{format_type}"
            filepath = Config.OUTPUT_DIR / filename
            mesh.export(str(filepath))
            output_files[format_type] = str(filepath)
        
        logger.info("3D model generated successfully")
        return output_files, scene_codes
        
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
        raise

def get_latest_model_file(format_type):
    """
    Get the most recently generated model file of specified format
    
    Args:
        format_type: File format ('obj', 'glb')
        
    Returns:
        Path: Path to latest model file or None
    """
    try:
        files = list(Config.OUTPUT_DIR.glob(f"model_*.{format_type}"))
        if not files:
            return None
        
        # Return the most recent file
        import os
        latest_file = max(files, key=os.path.getctime)
        return latest_file
        
    except Exception as e:
        logger.error(f"Error finding latest model file: {e}")
        return None

def cleanup_old_models(max_files=10):
    """
    Clean up old model files, keeping only the most recent ones
    
    Args:
        max_files: Maximum number of files to keep per format
    """
    try:
        import os
        
        for format_type in ['obj', 'glb']:
            files = list(Config.OUTPUT_DIR.glob(f"model_*.{format_type}"))
            if len(files) > max_files:
                # Sort by creation time and remove oldest
                files.sort(key=os.path.getctime)
                files_to_remove = files[:-max_files]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed old model file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
                        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
