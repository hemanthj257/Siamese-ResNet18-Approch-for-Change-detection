from firebase_config import db  # Changed from app.firebase_config

def test_connection():
    try:
        # Try to access Firestore
        users_ref = db.collection('users')
        print("Firebase connection successful!")
        return True
    except Exception as e:
        print(f"Firebase connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
