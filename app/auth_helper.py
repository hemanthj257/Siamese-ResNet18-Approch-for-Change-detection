import streamlit as st
from firebase_admin import auth
import pyrebase
import json
import os
import re

def is_valid_email(email):
    """Check if email format is valid"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

# Get the paths to config files
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'firebase_config.json')
cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'firebase_credentials.json')

# Load Firebase configuration
with open(config_path) as f:
    config = json.load(f)

# Update the serviceAccount path in config
config["serviceAccount"] = cred_path

# Initialize Pyrebase with error handling
try:
    firebase = pyrebase.initialize_app(config)
    pb_auth = firebase.auth()
except Exception as e:
    print(f"Firebase initialization error: {e}")
    print(f"Config path: {config_path}")
    print(f"Credentials path: {cred_path}")
    raise

def login_user(email: str, password: str):
    try:
        # Validate email format
        if not email or not is_valid_email(email):
            st.error("Please enter a valid email address")
            return False
        
        if not password:
            st.error("Please enter your password")
            return False

        # Authenticate with Pyrebase
        user = pb_auth.sign_in_with_email_and_password(email, password)
        
        # Get the Firebase user
        firebase_user = auth.get_user_by_email(email)
        
        # Store user info in session state with explicit keys
        st.session_state.user = {
            'uid': firebase_user.uid,
            'email': email,
            'token': user['idToken'],
            'display_name': firebase_user.display_name or email.split('@')[0]
        }
        return True
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.error("Invalid email format")
        elif "EMAIL_NOT_FOUND" in error_message:
            st.error("Email not registered")
        elif "INVALID_PASSWORD" in error_message:
            st.error("Incorrect password")
        else:
            st.error(f"Login failed: {error_message}")
        print(f"Login error: {e}")
        return False

def create_user(email: str, password: str):
    try:
        # Validate email format
        if not email or not is_valid_email(email):
            st.error("Please enter a valid email address")
            return False
        
        if not password or len(password) < 6:
            st.error("Password must be at least 6 characters long")
            return False

        # Create user with Pyrebase
        user = pb_auth.create_user_with_email_and_password(email, password)
        return True
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            st.error("Email already registered")
        elif "WEAK_PASSWORD" in error_message:
            st.error("Password should be at least 6 characters")
        else:
            st.error(f"Registration failed: {error_message}")
        print(f"Create user error: {e}")
        return False

def logout_user():
    """Clean logout that properly resets the session state"""
    try:
        # Clear the entire session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        return True
    except Exception as e:
        print(f"Logout error: {e}")
        return False

def is_logged_in():
    """Check if user is logged in"""
    return st.session_state.get('user') is not None

def get_current_user():
    """Safely get current user data"""
    return st.session_state.get('user', {})
