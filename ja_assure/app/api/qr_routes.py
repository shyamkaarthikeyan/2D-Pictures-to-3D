"""
QR code related API routes
"""

import base64
import logging
import socket
from flask import Blueprint, request, jsonify, render_template
from app.utils.session_manager import session_manager
from app.utils.image_processing import validate_image_format

logger = logging.getLogger(__name__)
qr_bp = Blueprint('qr', __name__)

@qr_bp.route('/qr-upload-session', methods=['POST'])
def create_qr_upload_session():
    """Create a unique upload session for QR code"""
    try:
        session_id = session_manager.create_session()
        
        # Get the actual host IP that the phone can reach
        host_ip = request.host.split(':')[0]
        if host_ip == 'localhost' or host_ip == '127.0.0.1':
            # Try to get the actual network IP
            try:
                # Connect to a remote address to determine local IP
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    host_ip = s.getsockname()[0]
            except:
                # Fallback: use the host from request headers
                host_ip = request.environ.get('HTTP_HOST', request.host).split(':')[0]
        
        # Use the port from the current request
        port = request.host.split(':')[1] if ':' in request.host else '5000'
        upload_url = f"http://{host_ip}:{port}/mobile-upload/{session_id}"
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'upload_url': upload_url
        })
        
    except Exception as e:
        logger.error(f"Error creating QR upload session: {e}")
        return jsonify({'error': str(e)}), 500

@qr_bp.route('/mobile-upload/<session_id>', methods=['POST'])
def handle_mobile_upload(session_id):
    """Handle image upload from mobile device"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid or expired session'}), 404
        
        uploaded_files = request.files.getlist('images')
        
        if not uploaded_files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        # Process uploaded images
        processed_images = []
        for file in uploaded_files[:4]:  # Limit to 4 images
            if file and file.filename and validate_image_format(file):
                file_content = file.read()
                file_b64 = base64.b64encode(file_content).decode()
                file_type = file.content_type or 'image/jpeg'
                data_url = f"data:{file_type};base64,{file_b64}"
                processed_images.append(data_url)
        
        # Update session with images
        success = session_manager.update_session(session_id, processed_images)
        if not success:
            return jsonify({'error': 'Failed to update session'}), 500
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(processed_images)} images',
            'image_count': len(processed_images)
        })
    
    except Exception as e:
        logger.error(f"Error handling mobile upload: {e}")
        return jsonify({'error': str(e)}), 500

@qr_bp.route('/check-qr-session/<session_id>')
def check_qr_session(session_id):
    """Check status of QR upload session"""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid or expired session'}), 404
        
        return jsonify({
            'success': True,
            'status': session['status'],
            'images': session.get('images', []),
            'image_count': len(session.get('images', []))
        })
    
    except Exception as e:
        logger.error(f"Error checking QR session: {e}")
        return jsonify({'error': str(e)}), 500
