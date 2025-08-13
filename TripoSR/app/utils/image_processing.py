"""
Image processing utilities
"""

import base64
import io
import numpy as np
from PIL import Image
import torch
import logging
from app.core.model_loader import get_rembg_session
from tsr.utils import remove_background, resize_foreground

logger = logging.getLogger(__name__)

def preprocess_image(image_data, do_remove_background=True, foreground_ratio=0.85):
    """
    Preprocess the input image
    
    Args:
        image_data: Base64 encoded image data
        do_remove_background: Whether to remove background
        foreground_ratio: Ratio for foreground sizing
        
    Returns:
        PIL.Image: Preprocessed image
    """
    try:
        # Decode base64 image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        
        def fill_background(img):
            """Fill transparent background with gray"""
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = img_array[:, :, :3] * img_array[:, :, 3:4] + (1 - img_array[:, :, 3:4]) * 0.5
            return Image.fromarray((img_array * 255.0).astype(np.uint8))

        if do_remove_background:
            image_rgb = image.convert("RGB")
            rembg_session = get_rembg_session()
            image = remove_background(image_rgb, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            if image.mode == "RGBA":
                image = fill_background(image)
            else:
                image = image.convert("RGB")
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def image_to_base64(image, format='PNG'):
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        str: Base64 encoded image
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{image_b64}"

def decode_base64_image(image_data):
    """
    Decode base64 image data to PIL Image
    
    Args:
        image_data: Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes))

def validate_image_format(file):
    """
    Validate uploaded image file format
    
    Args:
        file: Uploaded file object
        
    Returns:
        bool: True if valid format
    """
    if not file or not file.filename:
        return False
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'webp'}
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    return file_extension in allowed_extensions
