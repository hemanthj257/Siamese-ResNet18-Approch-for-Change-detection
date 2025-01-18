from firebase_config import db
import datetime
import json
import os
import shutil

def save_detection_session(user_id, img1_path, img2_path, result_mask):
    """
    Save detection session with images and result
    """
    try:
        if not user_id:
            raise ValueError("User ID is required")

        # Create results directory structure
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results', str(user_id))
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save paths
        saved_img1_path = os.path.join(results_dir, f'img1_{timestamp}.png')
        saved_img2_path = os.path.join(results_dir, f'img2_{timestamp}.png')
        saved_result_path = os.path.join(results_dir, f'result_{timestamp}.png')
        
        # Copy/save files
        shutil.copy2(img1_path, saved_img1_path)
        shutil.copy2(img2_path, saved_img2_path)
        
        # Save the result mask
        import cv2
        import numpy as np
        cv2.imwrite(saved_result_path, (result_mask * 255).astype(np.uint8))
        
        # Create relative paths for storage
        rel_img1_path = os.path.relpath(saved_img1_path, base_dir)
        rel_img2_path = os.path.relpath(saved_img2_path, base_dir)
        rel_result_path = os.path.relpath(saved_result_path, base_dir)
        
        # Save to Firestore
        session_ref = db.collection('users').document(str(user_id)).collection('sessions')
        session_data = {
            'timestamp': datetime.datetime.now(),
            'image1_path': rel_img1_path,
            'image2_path': rel_img2_path,
            'result_path': rel_result_path
        }
        session_ref.add(session_data)
        
        # Clean up temporary files
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
            
        return True
    except Exception as e:
        print(f"Error saving session: {e}")
        return False

def get_user_sessions(user_id):
    """
    Get user's previous sessions with full file paths
    """
    try:
        if not user_id:
            return []
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sessions = []
        
        # Get sessions from Firestore
        session_refs = db.collection('users').document(str(user_id)).collection('sessions')\
            .order_by('timestamp', direction='DESCENDING').stream()
        
        for session in session_refs:
            session_data = session.to_dict()
            # Convert relative paths to absolute
            session_data['image1_path'] = os.path.join(base_dir, session_data['image1_path'])
            session_data['image2_path'] = os.path.join(base_dir, session_data['image2_path'])
            session_data['result_path'] = os.path.join(base_dir, session_data['result_path'])
            
            # Only add if files exist
            if all(os.path.exists(p) for p in [session_data['image1_path'], 
                                             session_data['image2_path'], 
                                             session_data['result_path']]):
                sessions.append(session_data)
        
        return sessions
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return []
