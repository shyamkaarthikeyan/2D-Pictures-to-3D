"""
Main API routes for image processing and 3D generation
"""

import base64
import io
import logging
from flask import Blueprint, request, jsonify, send_file
from app.utils.image_processing import preprocess_image, image_to_base64, decode_base64_image
from app.utils.model_generation import generate_3d_model, get_latest_model_file

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

@api_bp.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and preprocessing"""
    try:
        data = request.json
        image_data = data.get('image')
        do_remove_background = data.get('removeBackground', True)
        foreground_ratio = data.get('foregroundRatio', 0.85)
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data, do_remove_background, foreground_ratio)
        
        # Convert processed image to base64
        processed_image_b64 = image_to_base64(processed_image)
        
        return jsonify({
            'success': True,
            'processedImage': processed_image_b64
        })
        
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/generate', methods=['POST'])
def generate_model():
    """Generate 3D model from processed image"""
    try:
        data = request.json
        image_data = data.get('processedImage')
        mc_resolution = data.get('mcResolution', 256)
        
        if not image_data:
            return jsonify({'error': 'No processed image data provided'}), 400
        
        # Decode the processed image
        image = decode_base64_image(image_data)
        
        # Generate 3D model
        output_files, scene_codes = generate_3d_model(image, mc_resolution)
        
        # Read files and encode as base64
        model_data = {}
        for format_type, filepath in output_files.items():
            with open(filepath, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode()
                model_data[format_type] = file_data
        
        return jsonify({
            'success': True,
            'models': model_data,
            'message': 'Model generated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error in generate_model: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/download/<model_format>')
def download_model(model_format):
    """Download generated model file"""
    try:
        # Find the most recent model file
        latest_file = get_latest_model_file(model_format)
        if not latest_file:
            return jsonify({'error': 'No model file found'}), 404
        
        return send_file(latest_file, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({'error': str(e)}), 500
