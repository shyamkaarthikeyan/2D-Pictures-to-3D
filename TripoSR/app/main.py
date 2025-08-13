"""Main routes"""

from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@main_bp.route('/mobile-upload/<session_id>')
def mobile_upload_page(session_id):
    """Mobile upload page for QR code scanning"""
    return render_template('mobile_upload.html', session_id=session_id)
