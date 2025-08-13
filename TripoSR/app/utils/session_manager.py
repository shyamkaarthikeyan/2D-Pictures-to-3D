"""
Session management utilities for QR code uploads
"""

import time
import uuid
import logging
from app.core.config import Config

logger = logging.getLogger(__name__)

class SessionManager:
    """Manage upload sessions for QR code functionality"""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self):
        """
        Create a new upload session
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'images': [],
            'created_at': time.time(),
            'status': 'waiting'
        }
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id):
        """
        Get session data by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            dict: Session data or None if not found/expired
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if time.time() - session['created_at'] > Config.SESSION_TIMEOUT:
            self.delete_session(session_id)
            return None
        
        return session
    
    def update_session(self, session_id, images):
        """
        Update session with uploaded images
        
        Args:
            session_id: Session identifier
            images: List of image data
            
        Returns:
            bool: True if updated successfully
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session['images'] = images[:Config.MAX_IMAGES_PER_SESSION]
        session['status'] = 'completed'
        session['updated_at'] = time.time()
        
        logger.info(f"Updated session {session_id} with {len(images)} images")
        return True
    
    def delete_session(self, session_id):
        """
        Delete a session
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['created_at'] > Config.SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Global session manager instance
session_manager = SessionManager()
