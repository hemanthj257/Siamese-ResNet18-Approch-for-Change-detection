import firebase_admin
from firebase_admin import credentials, auth, firestore
import os

# Get the absolute path to the credentials file
cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'firebase_credentials.json')

# Add error handling for file existence
if not os.path.exists(cred_path):
    raise FileNotFoundError(f"""
    Firebase credentials file not found at {cred_path}
    Please ensure you have:
    1. Downloaded the credentials from Firebase Console
    2. Created the config directory: mkdir -p /home/hehe/final/app/config
    3. Moved the credentials file to: /home/hehe/final/app/config/firebase_credentials.json
    """)

# Initialize Firebase
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()
