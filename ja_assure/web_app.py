import logging
import os
import tempfile
import time
import base64
import io
import zipfile
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import numpy as np
import rembg
import torch
from PIL import Image
import trimesh

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from tsr.bake_texture import bake_texture

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
if torch.cuda.is_available():
    device = "cuda:0"
    logger.info("Using CUDA device")
else:
    device = "cpu"
    logger.info("Using CPU device")

# Load TripoSR model
logger.info("Loading TripoSR model...")
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
logger.info("Model loaded successfully")

# Initialize background removal
rembg_session = rembg.new_session()

# Create output directory
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

def preprocess_image(image_data, do_remove_background=True, foreground_ratio=0.85):
    """Preprocess the input image"""
    try:
        # Decode base64 image
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        
        def fill_background(img):
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = img_array[:, :, :3] * img_array[:, :, 3:4] + (1 - img_array[:, :, 3:4]) * 0.5
            return Image.fromarray((img_array * 255.0).astype(np.uint8))

        if do_remove_background:
            image_rgb = image.convert("RGB")
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

def generate_3d_model(image, mc_resolution=256, formats=["obj", "glb"]):
    """Generate 3D model from preprocessed image"""
    try:
        logger.info("Generating 3D model...")
        
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
            filepath = output_dir / filename
            mesh.export(str(filepath))
            output_files[format_type] = str(filepath)
        
        logger.info("3D model generated successfully")
        return output_files, scene_codes
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
        raise

@app.route('/')
def index():
    """Serve the main application"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/upload', methods=['POST'])
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
        buffer = io.BytesIO()
        processed_image.save(buffer, format='PNG')
        processed_image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'processedImage': f"data:image/png;base64,{processed_image_b64}"
        })
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_model():
    """Generate 3D model from processed image"""
    try:
        data = request.json
        image_data = data.get('processedImage')
        mc_resolution = data.get('mcResolution', 256)
        
        if not image_data:
            return jsonify({'error': 'No processed image data provided'}), 400
        
        # Decode the processed image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
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

@app.route('/api/download/<model_format>')
def download_model(model_format):
    """Download generated model file"""
    try:
        # Find the most recent model file
        files = list(output_dir.glob(f"model_*.{model_format}"))
        if not files:
            return jsonify({'error': 'No model file found'}), 404
        
        latest_file = max(files, key=os.path.getctime)
        return send_file(latest_file, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({'error': str(e)}), 500

# QR Code upload endpoints
@app.route('/api/qr-upload-session', methods=['POST'])
def create_qr_upload_session():
    """Create a unique upload session for QR code"""
    try:
        session_id = str(uuid.uuid4())
        
        # Store session in memory (in production, use Redis or database)
        if not hasattr(app, 'upload_sessions'):
            app.upload_sessions = {}
        
        app.upload_sessions[session_id] = {
            'images': [],
            'created_at': time.time(),
            'status': 'waiting'
        }
        
        # Get the actual host IP that the phone can reach
        host_ip = request.host.split(':')[0]
        if host_ip == 'localhost' or host_ip == '127.0.0.1':
            # Try to get the actual network IP
            import socket
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

@app.route('/mobile-upload/<session_id>')
def mobile_upload_page(session_id):
    """Mobile upload page for QR code scanning"""
    return render_template_string(MOBILE_UPLOAD_TEMPLATE, session_id=session_id)

@app.route('/api/mobile-upload/<session_id>', methods=['POST'])
def handle_mobile_upload(session_id):
    """Handle image upload from mobile device"""
    try:
        if not hasattr(app, 'upload_sessions') or session_id not in app.upload_sessions:
            return jsonify({'error': 'Invalid session'}), 404
        
        uploaded_files = request.files.getlist('images')
        
        if not uploaded_files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        session_data = app.upload_sessions[session_id]
        
        # Process uploaded images
        processed_images = []
        for file in uploaded_files[:4]:  # Limit to 4 images
            if file and file.filename:
                file_content = file.read()
                file_b64 = base64.b64encode(file_content).decode()
                file_type = file.content_type or 'image/jpeg'
                data_url = f"data:{file_type};base64,{file_b64}"
                processed_images.append(data_url)
        
        session_data['images'] = processed_images
        session_data['status'] = 'completed'
        session_data['updated_at'] = time.time()
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(processed_images)} images',
            'image_count': len(processed_images)
        })
    
    except Exception as e:
        logger.error(f"Error handling mobile upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-qr-session/<session_id>')
def check_qr_session(session_id):
    """Check status of QR upload session"""
    try:
        if not hasattr(app, 'upload_sessions') or session_id not in app.upload_sessions:
            return jsonify({'error': 'Invalid session'}), 404
        
        session_data = app.upload_sessions[session_id]
        
        # Clean up old sessions (older than 1 hour)
        current_time = time.time()
        if current_time - session_data['created_at'] > 3600:
            del app.upload_sessions[session_id]
            return jsonify({'error': 'Session expired'}), 410
        
        return jsonify({
            'success': True,
            'status': session_data['status'],
            'images': session_data.get('images', []),
            'image_count': len(session_data.get('images', []))
        })
    
    except Exception as e:
        logger.error(f"Error checking QR session: {e}")
        return jsonify({'error': str(e)}), 500

# Mobile upload template
MOBILE_UPLOAD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Images - TripoSR</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { 
            font-family: 'Inter', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .upload-area { 
            border: 2px dashed #ccc; 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        .preview-image { width: 100%; height: 100px; object-fit: cover; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="max-w-md mx-auto p-4">
        <div class="bg-white/20 backdrop-blur-lg rounded-xl p-6 text-white">
            <h1 class="text-2xl font-bold mb-4 text-center">Upload Images for 3D Generation</h1>
            
            <div class="upload-area" id="upload-area">
                <svg class="mx-auto h-12 w-12 text-white/70 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p class="text-white mb-4">Select up to 4 images</p>
                <input type="file" id="file-input" multiple accept="image/*" class="hidden">
                <button onclick="document.getElementById('file-input').click()" class="bg-blue-500 text-white px-6 py-2 rounded-lg">
                    Choose Images
                </button>
            </div>
            
            <div id="preview-container" class="hidden">
                <p class="text-white mt-4 mb-2">Selected Images:</p>
                <div id="preview-grid" class="preview-grid"></div>
            </div>
            
            <button id="upload-btn" class="w-full bg-green-500 text-white py-3 rounded-lg mt-4 disabled:opacity-50" disabled>
                Upload Images
            </button>
            
            <div id="success-message" class="hidden bg-green-500 text-white p-4 rounded-lg mt-4 text-center">
                Images uploaded successfully! You can now close this page.
            </div>
        </div>
    </div>

    <script>
        const sessionId = '{{ session_id }}';
        const fileInput = document.getElementById('file-input');
        const previewContainer = document.getElementById('preview-container');
        const previewGrid = document.getElementById('preview-grid');
        const uploadBtn = document.getElementById('upload-btn');
        const successMessage = document.getElementById('success-message');
        
        let selectedFiles = [];

        fileInput.addEventListener('change', function(e) {
            selectedFiles = Array.from(e.target.files).slice(0, 4);
            displayPreviews();
            uploadBtn.disabled = selectedFiles.length === 0;
        });

        function displayPreviews() {
            previewGrid.innerHTML = '';
            previewContainer.classList.remove('hidden');
            
            selectedFiles.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.className = 'preview-image';
                    previewGrid.appendChild(img);
                };
                reader.readAsDataURL(file);
            });
        }

        uploadBtn.addEventListener('click', async function() {
            const formData = new FormData();
            selectedFiles.forEach(file => formData.append('images', file));
            
            try {
                uploadBtn.disabled = true;
                uploadBtn.textContent = 'Uploading...';
                
                const response = await fetch(`/api/mobile-upload/${sessionId}`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    successMessage.classList.remove('hidden');
                    previewContainer.classList.add('hidden');
                    uploadBtn.classList.add('hidden');
                } else {
                    alert('Upload failed: ' + result.error);
                }
            } catch (error) {
                alert('Upload failed: ' + error.message);
            }
        });
    </script>
</body>
</html>
"""

# HTML Template with integrated Nexio design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D Image to 3D</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/framer-motion@11.0.0/dist/framer-motion.min.js"></script>
    <script>
        // Load QRCode library with proper error handling
        window.QRCodeLoaded = false;
        const qrScript = document.createElement('script');
        qrScript.src = 'https://cdn.jsdelivr.net/npm/qrcode@1.5.3/build/qrcode.min.js';
        qrScript.onload = function() {
            window.QRCodeLoaded = true;
            console.log('QRCode library loaded successfully');
        };
        qrScript.onerror = function() {
            console.error('Failed to load QRCode library from CDN, trying alternative...');
            // Try alternative CDN
            const altScript = document.createElement('script');
            altScript.src = 'https://unpkg.com/qrcode@1.5.3/build/qrcode.min.js';
            altScript.onload = function() {
                window.QRCodeLoaded = true;
                console.log('QRCode library loaded from alternative CDN');
            };
            altScript.onerror = function() {
                console.error('Failed to load QRCode library from all sources');
            };
            document.head.appendChild(altScript);
        };
        document.head.appendChild(qrScript);
    </script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Inter', sans-serif; 
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            overflow-y: auto;
            background: #0a0a0f !important;
            color: white !important;
        }
        
        * {
            box-sizing: border-box;
        }
        
        html, body {
            background: #0a0a0f !important;
            min-height: 100vh;
            scroll-behavior: smooth;
        }
        
        /* Nexio-style background - exact match */
        .nexio-bg {
            background: #0a0a0f !important;
            background-image: 
                linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #7209b7 100%),
                radial-gradient(circle at 20% 20%, rgba(102, 126, 234, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(162, 28, 175, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 60%, rgba(59, 130, 246, 0.2) 0%, transparent 50%) !important;
            position: relative;
            min-height: 100vh;
            width: 100%;
        }
        
        /* Video-like animated overlay */
        .nexio-bg::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: 
                linear-gradient(45deg, 
                    rgba(102, 126, 234, 0.1) 0%, 
                    rgba(59, 130, 246, 0.15) 25%,
                    rgba(147, 51, 234, 0.1) 50%,
                    rgba(219, 39, 119, 0.15) 75%,
                    rgba(239, 68, 68, 0.1) 100%
                );
            animation: colorShift 8s ease-in-out infinite;
            mix-blend-mode: color-dodge;
            pointer-events: none;
            z-index: -2;
        }
        
        @keyframes colorShift {
            0%, 100% { 
                background: linear-gradient(45deg, 
                    rgba(102, 126, 234, 0.1) 0%, 
                    rgba(59, 130, 246, 0.15) 25%,
                    rgba(147, 51, 234, 0.1) 50%,
                    rgba(219, 39, 119, 0.15) 75%,
                    rgba(239, 68, 68, 0.1) 100%
                );
            }
            25% { 
                background: linear-gradient(45deg, 
                    rgba(59, 130, 246, 0.15) 0%, 
                    rgba(147, 51, 234, 0.1) 25%,
                    rgba(219, 39, 119, 0.15) 50%,
                    rgba(239, 68, 68, 0.1) 75%,
                    rgba(102, 126, 234, 0.1) 100%
                );
            }
            50% { 
                background: linear-gradient(45deg, 
                    rgba(147, 51, 234, 0.1) 0%, 
                    rgba(219, 39, 119, 0.15) 25%,
                    rgba(239, 68, 68, 0.1) 50%,
                    rgba(102, 126, 234, 0.1) 75%,
                    rgba(59, 130, 246, 0.15) 100%
                );
            }
            75% { 
                background: linear-gradient(45deg, 
                    rgba(219, 39, 119, 0.15) 0%, 
                    rgba(239, 68, 68, 0.1) 25%,
                    rgba(102, 126, 234, 0.1) 50%,
                    rgba(59, 130, 246, 0.15) 75%,
                    rgba(147, 51, 234, 0.1) 100%
                );
            }
        }
        
        /* Enhanced particle system */
        .nexio-bg::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background-image: 
                radial-gradient(circle 2px at 15% 25%, rgba(102, 126, 234, 0.6), transparent),
                radial-gradient(circle 1px at 85% 15%, rgba(147, 51, 234, 0.4), transparent),
                radial-gradient(circle 1px at 25% 75%, rgba(59, 130, 246, 0.5), transparent),
                radial-gradient(circle 2px at 75% 85%, rgba(219, 39, 119, 0.4), transparent),
                radial-gradient(circle 1px at 45% 35%, rgba(239, 68, 68, 0.3), transparent),
                radial-gradient(circle 1px at 65% 55%, rgba(102, 126, 234, 0.4), transparent),
                radial-gradient(circle 2px at 35% 65%, rgba(147, 51, 234, 0.5), transparent),
                radial-gradient(circle 1px at 55% 25%, rgba(59, 130, 246, 0.3), transparent);
            background-size: 300px 300px, 200px 200px, 400px 400px, 250px 250px, 350px 350px, 180px 180px, 320px 320px, 280px 280px;
            animation: particleFloat 15s linear infinite, particlePulse 3s ease-in-out infinite alternate;
            pointer-events: none;
            opacity: 0.8;
            z-index: -1;
        }
        
        @keyframes particleFloat {
            0% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(-10px, -15px) rotate(90deg); }
            50% { transform: translate(10px, -10px) rotate(180deg); }
            75% { transform: translate(-5px, 15px) rotate(270deg); }
            100% { transform: translate(0, 0) rotate(360deg); }
        }
        
        @keyframes particlePulse {
            0% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        
        .glass-effect {
            background: rgba(21, 22, 26, 0.7);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        /* Enhanced glassmorphism for panels - Nexio style */
        .enhanced-glass {
            background: rgba(21, 22, 26, 0.85) !important;
            backdrop-filter: blur(25px) saturate(180%) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.05) !important;
            position: relative;
        }
        
        .enhanced-glass::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.05) 0%, 
                rgba(147, 51, 234, 0.03) 50%, 
                rgba(219, 39, 119, 0.05) 100%
            );
            border-radius: inherit;
            pointer-events: none;
            z-index: -1;
        }
        
        .model-preview {
            border: 2px solid rgba(147, 51, 234, 0.6);
            border-radius: 12px;
            background: rgba(21, 22, 26, 0.4);
            box-shadow: 
                0 0 40px rgba(147, 51, 234, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .loading-spinner {
            border: 3px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            border-top: 3px solid #9333ea;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background-color: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #9333ea, #db2777);
            border-radius: 2px;
            transition: width 0.3s ease;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.6);
            animation: progressGlow 2s ease-in-out infinite alternate;
        }
        
        @keyframes progressGlow {
            0% { box-shadow: 0 0 15px rgba(102, 126, 234, 0.6); }
            100% { box-shadow: 0 0 25px rgba(147, 51, 234, 0.8); }
        }
        
        /* Nexio-style gradient text */
        .gradient-text {
            background: linear-gradient(135deg, 
                #667eea 0%, 
                #9333ea 30%, 
                #db2777 60%, 
                #f59e0b 100%
            ) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            animation: gradientShift 4s ease-in-out infinite;
            display: inline-block;
        }
        
        @keyframes gradientShift {
            0%, 100% { 
                background: linear-gradient(135deg, #667eea 0%, #9333ea 30%, #db2777 60%, #f59e0b 100%);
            }
            25% { 
                background: linear-gradient(135deg, #9333ea 0%, #db2777 30%, #f59e0b 60%, #667eea 100%);
            }
            50% { 
                background: linear-gradient(135deg, #db2777 0%, #f59e0b 30%, #667eea 60%, #9333ea 100%);
            }
            75% { 
                background: linear-gradient(135deg, #f59e0b 0%, #667eea 30%, #9333ea 60%, #db2777 100%);
            }
        }
        
        /* Enhanced hover effects - Nexio style */
        .glass-hover:hover {
            background: rgba(21, 22, 26, 0.95);
            border-color: rgba(147, 51, 234, 0.4);
            transform: translateY(-4px) scale(1.02);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.6),
                0 0 40px rgba(147, 51, 234, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .glass-hover:hover::before {
            background: linear-gradient(135deg, 
                rgba(102, 126, 234, 0.1) 0%, 
                rgba(147, 51, 234, 0.08) 50%, 
                rgba(219, 39, 119, 0.1) 100%
            );
        }
        
        /* Button enhancements */
        .nexio-button {
            background: linear-gradient(135deg, #667eea 0%, #9333ea 100%);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 
                0 4px 15px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .nexio-button:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #7c3aed 100%);
            transform: translateY(-2px);
            box-shadow: 
                0 8px 25px rgba(102, 126, 234, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        
        .nexio-button:active {
            transform: translateY(0);
            box-shadow: 
                0 4px 15px rgba(102, 126, 234, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a67d8, #6b46c1);
        }
    </style>
</head>
<body class="nexio-bg text-white">

    <!-- Navigation -->
    <nav class="relative z-50 flex justify-center p-4">
        <div class="enhanced-glass px-6 py-4 rounded-2xl flex justify-between items-center w-full max-w-6xl glass-hover">
            <div class="flex items-center gap-4">
                <h1 class="text-white text-xl font-bold gradient-text">2D Image to 3D</h1>
                <!-- Removed divider and AI-Powered 3D Generation from navbar -->
            </div>
            <button class="nexio-button px-6 py-2 text-white rounded-full font-medium">
                Generate 3D
            </button>
        </div>
    </nav>

    <!-- Hero Section -->
    <div class="container mx-auto px-4 py-16 text-center">
        <div class="max-w-4xl mx-auto space-y-8">
            <h1 class="text-5xl md:text-7xl font-bold leading-tight text-white">
                Transform Images to
                <span class="gradient-text">
                    3D Models
                </span>
            </h1>
            <p class="text-xl md:text-2xl text-white max-w-2xl mx-auto">
                Generate high-quality 3D models from a single image using state-of-the-art. Fast, accurate, and easy to use.
            </p>
        </div>
    </div>

    <!-- Main Application -->
    <div class="container mx-auto px-4 py-8">
        <div class="max-w-7xl mx-auto">
            <div class="grid lg:grid-cols-2 gap-8">
                
                <!-- Left Panel - Input Controls -->
                <div class="space-y-6">

                    <div class="enhanced-glass glass-hover p-6 rounded-2xl">
                        <h2 class="text-2xl font-semibold mb-4 text-white">Upload Images</h2>
                        <p class="text-white/60 text-sm mb-4">Upload 4 images (only the first image will be processed for 3D generation)</p>
                        
                        <!-- QR Code Upload Option -->
                        <div class="mb-4">
                            <div class="enhanced-glass p-4 rounded-xl">
                                <div class="flex items-center justify-between mb-3">
                                    <h3 class="text-lg font-medium text-white">Quick Upload via Phone</h3>
                                    <button id="generate-qr-btn" class="nexio-button px-4 py-2 text-sm">
                                        Scan QR Code
                                    </button>
                                </div>
                                
                                <div id="qr-code-container" class="hidden text-center">
                                    <div class="bg-white p-4 rounded-lg inline-block mb-3">
                                        <canvas id="qr-code-canvas" width="200" height="200"></canvas>
                                    </div>
                                    <p class="text-white/80 text-sm mb-2">Scan with your phone to upload images</p>
                                    <div id="qr-status" class="text-center">
                                        <div class="inline-flex items-center gap-2 text-blue-400">
                                            <div class="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                                            <span class="text-sm">Waiting for upload...</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Image Upload Grid -->
                        <div class="grid grid-cols-2 gap-4 mb-4">
                            <!-- Image 1 (Primary) -->
                            <div class="image-upload-slot" data-slot="1">
                                <div class="upload-area border-2 border-dashed border-blue-400/50 rounded-xl p-4 text-center cursor-pointer hover:border-blue-400/70 transition-colors bg-blue-500/10">
                                    <div class="upload-content">
                                        <svg class="mx-auto h-8 w-8 text-blue-400 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                        <p class="text-blue-400 text-sm font-medium">Image 1 (Primary)</p>
                                        <p class="text-white/60 text-xs">Click to upload</p>
                                    </div>
                                    <img class="preview-image hidden w-full h-20 object-cover rounded">
                                </div>
                                <input type="file" class="image-input hidden" accept="image/*" data-slot="1">
                            </div>
                            
                            <!-- Image 2 (Dummy) -->
                            <div class="image-upload-slot" data-slot="2">
                                <div class="upload-area border-2 border-dashed border-white/30 rounded-xl p-4 text-center cursor-pointer hover:border-white/50 transition-colors">
                                    <div class="upload-content">
                                        <svg class="mx-auto h-8 w-8 text-white/50 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                        <p class="text-white/70 text-sm">Image 2 (Optional)</p>
                                        <p class="text-white/60 text-xs">Click to upload</p>
                                    </div>
                                    <img class="preview-image hidden w-full h-20 object-cover rounded">
                                </div>
                                <input type="file" class="image-input hidden" accept="image/*" data-slot="2">
                            </div>
                            
                            <!-- Image 3 (Dummy) -->
                            <div class="image-upload-slot" data-slot="3">
                                <div class="upload-area border-2 border-dashed border-white/30 rounded-xl p-4 text-center cursor-pointer hover:border-white/50 transition-colors">
                                    <div class="upload-content">
                                        <svg class="mx-auto h-8 w-8 text-white/50 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                        <p class="text-white/70 text-sm">Image 3 (Optional)</p>
                                        <p class="text-white/60 text-xs">Click to upload</p>
                                    </div>
                                    <img class="preview-image hidden w-full h-20 object-cover rounded">
                                </div>
                                <input type="file" class="image-input hidden" accept="image/*" data-slot="3">
                            </div>
                            
                            <!-- Image 4 (Dummy) -->
                            <div class="image-upload-slot" data-slot="4">
                                <div class="upload-area border-2 border-dashed border-white/30 rounded-xl p-4 text-center cursor-pointer hover:border-white/50 transition-colors">
                                    <div class="upload-content">
                                        <svg class="mx-auto h-8 w-8 text-white/50 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                                            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                        <p class="text-white/70 text-sm">Image 4 (Optional)</p>
                                        <p class="text-white/60 text-xs">Click to upload</p>
                                    </div>
                                    <img class="preview-image hidden w-full h-20 object-cover rounded">
                                </div>
                                <input type="file" class="image-input hidden" accept="image/*" data-slot="4">
                            </div>
                        </div>
                        
                        <!-- Processed Image Preview -->
                        <div id="processed-preview" class="hidden mt-4">
                            <h3 class="text-lg font-medium mb-2 text-white">Processed Image (Primary)</h3>
                            <img id="processed-image" class="w-full rounded-lg border border-white/20">
                        </div>
                    </div>

                    <!-- Settings Panel -->
                    <div class="enhanced-glass glass-hover p-6 rounded-2xl">
                        <h2 class="text-2xl font-semibold mb-4 text-white">Settings</h2>
                        
                        <!-- Remove Background -->
                        <div class="mb-4">
                            <label class="flex items-center space-x-3">
                                <input type="checkbox" id="remove-bg" checked class="w-4 h-4 text-purple-600 bg-transparent border-2 border-white/30 rounded focus:ring-purple-500">
                                <span class="text-white">Remove Background</span>
                            </label>
                        </div>
                        
                        <!-- Foreground Ratio -->
                        <div class="mb-4">
                            <label class="block text-white mb-2">Foreground Ratio: <span id="ratio-value">0.85</span></label>
                            <input type="range" id="foreground-ratio" min="0.5" max="1.0" step="0.05" value="0.85" class="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer">
                        </div>
                        
                        <!-- MC Resolution -->
                        <div class="mb-6">
                            <label class="block text-white mb-2">Mesh Resolution: <span id="resolution-value">256</span></label>
                            <input type="range" id="mc-resolution" min="32" max="320" step="32" value="256" class="w-full h-2 bg-white/20 rounded-lg appearance-none cursor-pointer">
                        </div>
                        
                        <!-- Generate Button -->
                        <button id="generate-btn" disabled class="nexio-button w-full py-3 text-white rounded-xl font-semibold disabled:opacity-50 disabled:cursor-not-allowed">
                            Generate 3D Model
                        </button>
                        
                        <!-- Progress Bar -->
                        <div id="progress-container" class="hidden mt-4">
                            <div class="progress-bar">
                                <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
                            </div>
                            <p id="progress-text" class="text-center text-white/80 mt-2">Processing...</p>
                        </div>
                    </div>
                </div>

                <!-- Right Panel - 3D Model Display -->
                <div class="space-y-6">
                    <div class="enhanced-glass glass-hover p-6 rounded-2xl">
                        <h2 class="text-2xl font-semibold mb-4 text-white">3D Model Preview</h2>
                        
                        <!-- Model Viewer -->
                        <div id="model-container" class="aspect-square bg-black/20 rounded-xl flex items-center justify-center border border-white/20">
                            <div id="model-placeholder" class="text-center text-white/60">
                                <svg class="mx-auto h-16 w-16 mb-4" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clip-rule="evenodd"/>
                                </svg>
                                <p>Upload an image to generate 3D model</p>
                            </div>
                            
                            <!-- Loading State -->
                            <div id="loading-state" class="hidden text-center">
                                <div class="loading-spinner mx-auto mb-4"></div>
                                <p class="text-white/80">Generating 3D model...</p>
                            </div>
                            
                            <!-- Model Display -->
                            <div id="model-display" class="hidden w-full h-full">
                                <canvas id="model-canvas" class="w-full h-full rounded-xl"></canvas>
                            </div>
                        </div>
                        
                        <!-- Download Buttons -->
                        <div id="download-section" class="hidden mt-6 space-y-3">
                            <h3 class="text-lg font-medium">Download Model</h3>
                            <div class="flex gap-3">
                                <button id="download-obj" class="flex-1 py-2 px-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
                                    Download OBJ
                                </button>
                                <button id="download-glb" class="flex-1 py-2 px-4 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
                                    Download GLB
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Info Panel (About TripoSR section removed) -->
                        <div class="mt-4 space-y-2">
                            <div class="flex justify-between text-sm">
                                <span class="text-white/60">Speed:</span>
                                <span class="text-green-400">&lt; 0.5s on GPU</span>
                            </div>
                            <div class="flex justify-between text-sm">
                                <span class="text-white/60">Quality:</span>
                                <span class="text-blue-400">High Resolution</span>
                            </div>
                            <div class="flex justify-between text-sm">
                                <span class="text-white/60">Formats:</span>
                                <span class="text-purple-400">OBJ, GLB</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentImages = [null, null, null, null]; // Array for 4 images
        let processedImage = null;
        let generatedModels = null;
        let currentQRSession = null;
        let qrCheckInterval = null;

        // DOM elements
        const imageUploadSlots = document.querySelectorAll('.image-upload-slot');
        const processedPreview = document.getElementById('processed-preview');
        const processedImageEl = document.getElementById('processed-image');
        const removeBgCheckbox = document.getElementById('remove-bg');
        const foregroundRatioSlider = document.getElementById('foreground-ratio');
        const mcResolutionSlider = document.getElementById('mc-resolution');
        const ratioValue = document.getElementById('ratio-value');
        const resolutionValue = document.getElementById('resolution-value');
        const generateBtn = document.getElementById('generate-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const modelPlaceholder = document.getElementById('model-placeholder');
        const loadingState = document.getElementById('loading-state');
        const modelDisplay = document.getElementById('model-display');
        const downloadSection = document.getElementById('download-section');
        const downloadObjBtn = document.getElementById('download-obj');
        const downloadGlbBtn = document.getElementById('download-glb');
        
        // QR code elements
        const generateQRBtn = document.getElementById('generate-qr-btn');
        const qrCodeContainer = document.getElementById('qr-code-container');
        const qrCodeCanvas = document.getElementById('qr-code-canvas');
        const qrStatus = document.getElementById('qr-status');

        // Event listeners for image upload slots
        imageUploadSlots.forEach(slot => {
            const uploadArea = slot.querySelector('.upload-area');
            const imageInput = slot.querySelector('.image-input');
            const previewImage = slot.querySelector('.preview-image');
            const uploadContent = slot.querySelector('.upload-content');
            const slotNumber = parseInt(slot.dataset.slot);

            uploadArea.addEventListener('click', () => imageInput.click());
            uploadArea.addEventListener('dragover', (e) => handleDragOver(e, uploadArea));
            uploadArea.addEventListener('drop', (e) => handleDrop(e, uploadArea, slotNumber));
            imageInput.addEventListener('change', (e) => handleImageSelect(e, slotNumber, previewImage, uploadContent));
        });

        // Other event listeners
        foregroundRatioSlider.addEventListener('input', updateRatioValue);
        mcResolutionSlider.addEventListener('input', updateResolutionValue);
        generateBtn.addEventListener('click', generateModel);
        downloadObjBtn.addEventListener('click', () => downloadModel('obj'));
        downloadGlbBtn.addEventListener('click', () => downloadModel('glb'));
        generateQRBtn.addEventListener('click', generateQRCode);

        function handleDragOver(e, uploadArea) {
            e.preventDefault();
            uploadArea.classList.add('border-white/70');
        }

        function handleDrop(e, uploadArea, slotNumber) {
            e.preventDefault();
            uploadArea.classList.remove('border-white/70');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0], slotNumber);
            }
        }

        function handleImageSelect(e, slotNumber, previewImage, uploadContent) {
            const file = e.target.files[0];
            if (file) {
                handleFile(file, slotNumber, previewImage, uploadContent);
            }
        }

        function handleFile(file, slotNumber, previewImage, uploadContent) {
            if (!file.type.startsWith('image/')) {
                alert('Please select a valid image file');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                currentImages[slotNumber - 1] = e.target.result; // Store in array (0-indexed)
                
                // Update preview for this slot
                const slot = document.querySelector(`[data-slot="${slotNumber}"]`);
                const previewImg = slot.querySelector('.preview-image');
                const uploadContent = slot.querySelector('.upload-content');
                
                previewImg.src = e.target.result;
                previewImg.classList.remove('hidden');
                uploadContent.classList.add('hidden');
                
                // Only process the first image for 3D generation
                if (slotNumber === 1) {
                    processImage();
                }
                
                // Update generate button state
                updateGenerateButtonState();
            };
            reader.readAsDataURL(file);
        }

        function updateGenerateButtonState() {
            // Enable generate button only if first image is uploaded
            generateBtn.disabled = !currentImages[0];
        }

        async function processImage() {
            // Only process the first image
            if (!currentImages[0]) return;

            try {
                showProgress('Processing primary image...', 25);
                
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: currentImages[0], // Only use the first image
                        removeBackground: removeBgCheckbox.checked,
                        foregroundRatio: parseFloat(foregroundRatioSlider.value)
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    processedImage = result.processedImage;
                    processedImageEl.src = processedImage;
                    processedPreview.classList.remove('hidden');
                    showProgress('Primary image processed successfully!', 100);
                    setTimeout(hideProgress, 1000);
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Error processing image:', error);
                alert('Error processing primary image: ' + error.message);
                hideProgress();
            }
        }

        async function generateModel() {
            if (!processedImage) return;

            try {
                generateBtn.disabled = true;
                showModelLoading();
                showProgress('Generating 3D model...', 0);

                // Simulate progress updates
                const progressSteps = [
                    { text: 'Analyzing image...', progress: 20 },
                    { text: 'Creating triplane representation...', progress: 40 },
                    { text: 'Generating mesh...', progress: 70 },
                    { text: 'Optimizing geometry...', progress: 90 },
                    { text: 'Finalizing model...', progress: 100 }
                ];

                for (let i = 0; i < progressSteps.length; i++) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    showProgress(progressSteps[i].text, progressSteps[i].progress);
                }

                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        processedImage: processedImage,
                        mcResolution: parseInt(mcResolutionSlider.value)
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    generatedModels = result.models;
                    displayModel();
                    downloadSection.classList.remove('hidden');
                    hideProgress();
                    showSuccessMessage();
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Error generating model:', error);
                alert('Error generating model: ' + error.message);
                hideProgress();
                hideModelLoading();
            } finally {
                generateBtn.disabled = false;
            }
        }

        function displayModel() {
            hideModelLoading();
            modelDisplay.classList.remove('hidden');
            
            // Create a simple 3D preview placeholder
            const canvas = document.getElementById('model-canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Draw a simple 3D cube wireframe as placeholder
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#764ba2';
            ctx.lineWidth = 2;
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const size = 100;
            
            // Draw wireframe cube
            ctx.beginPath();
            // Front face
            ctx.rect(centerX - size/2, centerY - size/2, size, size);
            // Back face (offset)
            ctx.rect(centerX - size/2 + 20, centerY - size/2 - 20, size, size);
            // Connect corners
            ctx.moveTo(centerX - size/2, centerY - size/2);
            ctx.lineTo(centerX - size/2 + 20, centerY - size/2 - 20);
            ctx.moveTo(centerX + size/2, centerY - size/2);
            ctx.lineTo(centerX + size/2 + 20, centerY - size/2 - 20);
            ctx.moveTo(centerX - size/2, centerY + size/2);
            ctx.lineTo(centerX - size/2 + 20, centerY + size/2 - 20);
            ctx.moveTo(centerX + size/2, centerY + size/2);
            ctx.lineTo(centerX + size/2 + 20, centerY + size/2 - 20);
            ctx.stroke();
            
            // Add success text
            ctx.fillStyle = '#ffffff';
            ctx.font = '16px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('3D Model Generated Successfully!', centerX, centerY + size + 30);
            ctx.fillText('Click download buttons to save', centerX, centerY + size + 50);
        }

        function downloadModel(format) {
            if (!generatedModels || !generatedModels[format]) {
                alert('Model not available for download');
                return;
            }

            // Create download link
            const modelData = generatedModels[format];
            const byteCharacters = atob(modelData);
            const byteNumbers = new Array(byteCharacters.length);
            
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'application/octet-stream' });
            
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `triposr_model.${format}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function showProgress(text, progress) {
            progressContainer.classList.remove('hidden');
            progressText.textContent = text;
            progressFill.style.width = progress + '%';
        }

        function hideProgress() {
            progressContainer.classList.add('hidden');
        }

        function showModelLoading() {
            modelPlaceholder.classList.add('hidden');
            loadingState.classList.remove('hidden');
            modelDisplay.classList.add('hidden');
        }

        function hideModelLoading() {
            loadingState.classList.add('hidden');
        }

        function showSuccessMessage() {
            // Create and show success notification
            const notification = document.createElement('div');
            notification.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
            notification.textContent = '3D Model generated successfully!';
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 3000);
        }

        // Fallback QR code generator using Google Charts API
        function generateQRCodeFallback(text, canvas) {
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.onerror = function() {
                // If even the fallback fails, draw a simple message
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#ffffff';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#000000';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('QR Code Generation Failed', canvas.width/2, canvas.height/2 - 10);
                ctx.fillText('Please refresh and try again', canvas.width/2, canvas.height/2 + 10);
            };
            const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(text)}`;
            img.src = qrUrl;
        }

        async function generateQRCode() {
            try {
                generateQRBtn.disabled = true;
                generateQRBtn.textContent = 'Generating...';
                
                // Create upload session
                const response = await fetch('/api/qr-upload-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentQRSession = result.session_id;
                    
                    // Try to generate QR code with library, fallback to API if needed
                    try {
                        if (typeof QRCode !== 'undefined') {
                            await QRCode.toCanvas(qrCodeCanvas, result.upload_url, {
                                width: 200,
                                margin: 2,
                                color: {
                                    dark: '#000000',
                                    light: '#FFFFFF'
                                }
                            });
                        } else {
                            // Use fallback method
                            generateQRCodeFallback(result.upload_url, qrCodeCanvas);
                        }
                    } catch (qrError) {
                        console.log('Primary QR method failed, using fallback:', qrError);
                        generateQRCodeFallback(result.upload_url, qrCodeCanvas);
                    }
                    
                    qrCodeContainer.classList.remove('hidden');
                    generateQRBtn.textContent = 'Regenerate QR';
                    generateQRBtn.disabled = false;
                    
                    // Start checking for uploads
                    startQRStatusCheck();
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                console.error('Error generating QR code:', error);
                alert('Error generating QR code: ' + error.message);
                generateQRBtn.disabled = false;
                generateQRBtn.textContent = 'Scan QR Code';
            }
        }

        function startQRStatusCheck() {
            if (qrCheckInterval) clearInterval(qrCheckInterval);
            
            qrCheckInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/check-qr-session/${currentQRSession}`);
                    const result = await response.json();
                    
                    if (result.success) {
                        if (result.status === 'completed' && result.images.length > 0) {
                            // Images received via QR code
                            handleQRUploadComplete(result.images);
                        } else {
                            // Update status
                            updateQRStatus('waiting', result.image_count || 0);
                        }
                    } else if (response.status === 410) {
                        // Session expired
                        updateQRStatus('expired');
                        clearInterval(qrCheckInterval);
                    }
                } catch (error) {
                    console.error('Error checking QR status:', error);
                }
            }, 2000); // Check every 2 seconds
        }

        function handleQRUploadComplete(images) {
            clearInterval(qrCheckInterval);
            
            // Populate image slots with uploaded images (first image goes to slot 1, others to remaining slots)
            images.forEach((imageData, index) => {
                if (index < 4) {
                    currentImages[index] = imageData;
                    updateImageSlot(index + 1, imageData);
                }
            });
            
            // Process the first image if available
            if (images[0]) {
                processImage();
            }
            
            updateQRStatus('completed', images.length);
            updateGenerateButtonState();
            
            // Show success notification
            showNotification(`Successfully received ${images.length} images from phone!`, 'success');
        }

        function updateImageSlot(slotNumber, imageData) {
            const slot = document.querySelector(`[data-slot="${slotNumber}"]`);
            const previewImg = slot.querySelector('.preview-image');
            const uploadContent = slot.querySelector('.upload-content');
            
            previewImg.src = imageData;
            previewImg.classList.remove('hidden');
            uploadContent.classList.add('hidden');
        }

        function updateQRStatus(status, imageCount = 0) {
            const statusEl = document.getElementById('qr-status');
            
            switch (status) {
                case 'waiting':
                    statusEl.innerHTML = `
                        <div class="inline-flex items-center gap-2 text-blue-400">
                            <div class="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                            <span class="text-sm">Waiting for upload... ${imageCount > 0 ? `(${imageCount} received)` : ''}</span>
                        </div>
                    `;
                    break;
                case 'completed':
                    statusEl.innerHTML = `
                        <div class="inline-flex items-center gap-2 text-green-400">
                            <div class="w-2 h-2 bg-green-400 rounded-full"></div>
                            <span class="text-sm">Upload completed! (${imageCount} images)</span>
                        </div>
                    `;
                    break;
                case 'expired':
                    statusEl.innerHTML = `
                        <div class="inline-flex items-center gap-2 text-red-400">
                            <div class="w-2 h-2 bg-red-400 rounded-full"></div>
                            <span class="text-sm">Session expired. Generate new QR code.</span>
                        </div>
                    `;
                    break;
            }
        }

        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            const bgColor = type === 'success' ? 'bg-green-500' : type === 'error' ? 'bg-red-500' : 'bg-blue-500';
            notification.className = `fixed top-4 right-4 ${bgColor} text-white px-6 py-3 rounded-lg shadow-lg z-50`;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }

        // Clean up QR check interval when page unloads
        window.addEventListener('beforeunload', () => {
            if (qrCheckInterval) clearInterval(qrCheckInterval);
        });

        function updateRatioValue() {
            ratioValue.textContent = foregroundRatioSlider.value;
            if (currentImages[0]) { // Only reprocess if first image exists
                processImage(); // Reprocess with new ratio
            }
        }

        function updateResolutionValue() {
            resolutionValue.textContent = mcResolutionSlider.value;
        }

        // Add some visual effects - Enhanced Nexio style
        document.addEventListener('DOMContentLoaded', () => {
            // Wait for QRCode library to load
            function waitForQRCode() {
                if (typeof QRCode !== 'undefined') {
                    console.log('QRCode library loaded successfully');
                } else {
                    console.log('Waiting for QRCode library...');
                    setTimeout(waitForQRCode, 100);
                }
            }
            waitForQRCode();
            
            // Animate elements on scroll with stagger effect
            const observer = new IntersectionObserver((entries) => {
                entries.forEach((entry, index) => {
                    if (entry.isIntersecting) {
                        setTimeout(() => {
                            entry.target.style.opacity = '1';
                            entry.target.style.transform = 'translateY(0) scale(1)';
                            entry.target.classList.add('animate-in');
                        }, index * 150);
                    }
                });
            }, {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            });

            document.querySelectorAll('.enhanced-glass').forEach((el, index) => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(40px) scale(0.95)';
                el.style.transition = 'opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1), transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                observer.observe(el);
            });
            
            // Add floating animation to hero elements
            const heroTitle = document.querySelector('h1');
            if (heroTitle) {
                heroTitle.style.animation = 'float 6s ease-in-out infinite';
            }
            
            // Enhanced generate button interactions
            const generateBtn = document.getElementById('generate-btn');
            if (generateBtn) {
                generateBtn.addEventListener('mouseenter', () => {
                    generateBtn.style.transform = 'translateY(-3px) scale(1.05)';
                    generateBtn.style.boxShadow = '0 12px 35px rgba(102, 126, 234, 0.7)';
                });
                generateBtn.addEventListener('mouseleave', () => {
                    generateBtn.style.transform = 'translateY(0) scale(1)';
                    generateBtn.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.4)';
                });
            }
            
            // Add parallax effect to background
            let mouseX = 0, mouseY = 0;
            document.addEventListener('mousemove', (e) => {
                mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
                mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
                
                const bg = document.querySelector('.nexio-bg');
                if (bg) {
                    bg.style.transform = `translate(${mouseX * 5}px, ${mouseY * 5}px)`;
                }
            });
            
            // Add glow effect to interactive elements
            document.querySelectorAll('.enhanced-glass').forEach(el => {
                el.addEventListener('mouseenter', () => {
                    el.style.boxShadow = `
                        0 20px 60px rgba(0, 0, 0, 0.6),
                        0 0 60px rgba(147, 51, 234, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2)
                    `;
                });
                el.addEventListener('mouseleave', () => {
                    el.style.boxShadow = `
                        0 8px 32px rgba(0, 0, 0, 0.4),
                        inset 0 1px 0 rgba(255, 255, 255, 0.15),
                        0 0 0 1px rgba(255, 255, 255, 0.05)
                    `;
                });
            });
        });
        
        // Add enhanced animation keyframes
        const enhancedStyle = document.createElement('style');
        enhancedStyle.textContent = `
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                33% { transform: translateY(-8px) rotate(1deg); }
                66% { transform: translateY(4px) rotate(-1deg); }
            }
            
            @keyframes glow {
                0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
                50% { box-shadow: 0 0 60px rgba(147, 51, 234, 0.6); }
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.8; }
            }
            
            .animate-in {
                animation: glow 2s ease-in-out infinite alternate;
            }
            
            /* Enhanced transitions for all interactive elements */
            * {
                transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
        `;
        document.head.appendChild(enhancedStyle);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='TripoSR Web Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting TripoSR Web Application on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
