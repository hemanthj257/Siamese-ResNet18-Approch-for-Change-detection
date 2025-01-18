import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from test import quick_detect
from PIL import Image
import numpy as np
from auth_helper import login_user, create_user, logout_user  # Changed imports
from db_helper import save_detection_session, get_user_sessions  # Changed imports

# Set page config with custom width
st.set_page_config(page_title="Change Detection App", layout="wide")

# Add custom CSS to control image size
st.markdown("""
    <style>
        .stImage > img {
            max-width: 400px;
        }
    </style>
""", unsafe_allow_html=True)

def login_page():
    st.subheader("Login")
    
    # Add example format
    st.markdown("Enter your email and password to login")
    st.markdown("*Email format: example@domain.com*")
    
    with st.form("login_form"):
        email = st.text_input("Email").strip()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if login_user(email, password):
                st.success("Logged in successfully!")
                st.rerun()  # Changed from experimental_rerun
    
    st.markdown("---")
    st.subheader("Sign Up")
    with st.form("signup_form"):
        new_email = st.text_input("Email").strip()
        new_password = st.text_input("Password", type="password", 
                                   help="Password must be at least 6 characters long")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match!")
            else:
                if create_user(new_email, new_password):
                    st.success("Account created! Please login.")

def show_history():
    # Get user data safely
    user_data = st.session_state.get('user', {})
    user_id = user_data.get('uid')
    
    if not user_id:
        st.error("User session not found. Please login again.")
        return
        
    sessions = get_user_sessions(user_id)
    if sessions:
        st.subheader("Previous Detection Sessions")
        for session in sessions:
            with st.expander(f"Session from {session['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(session['image1_path'], caption="First Image", width=200)
                with col2:
                    st.image(session['image2_path'], caption="Second Image", width=200)
                with col3:
                    st.image(session['result_path'], caption="Result", width=200)

def main():
    try:
        st.title("Change Detection System")

        # Initialize session state if it doesn't exist
        if 'user' not in st.session_state:
            st.session_state.user = None

        # Show logout button if logged in
        if st.session_state.user is not None:
            if st.sidebar.button("Logout"):
                logout_user()
                st.rerun()

        # Check if user is logged in
        if st.session_state.user is None:
            login_page()
            return

        # Rest of the app (only shown when logged in)
        tab1, tab2 = st.tabs(["Detect Changes", "History"])

        with tab1:
            st.write("Upload two images to detect changes between them")

            # Create three columns for better spacing
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("First Image (Current)")
                img1_file = st.file_uploader("Upload first image", type=['png', 'jpg', 'jpeg'], key="img1")
                if img1_file:
                    img1 = Image.open(img1_file).convert('RGB')
                    # Resize image before displaying
                    display_img1 = img1.copy()
                    display_img1.thumbnail((400, 400))
                    st.image(display_img1, caption="First Image")

            with col2:
                st.subheader("Second Image (Past)")
                img2_file = st.file_uploader("Upload second image", type=['png', 'jpg', 'jpeg'], key="img2")
                if img2_file:
                    img2 = Image.open(img2_file).convert('RGB')
                    # Resize image before displaying
                    display_img2 = img2.copy()
                    display_img2.thumbnail((400, 400))
                    st.image(display_img2, caption="Second Image")

            with col3:
                if img1_file and img2_file:
                    st.subheader("Change Detection Result")
                    if st.button("Detect Changes"):
                        with st.spinner("Processing..."):
                            # Save uploaded files temporarily
                            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
                            os.makedirs(temp_dir, exist_ok=True)
                            
                            img1_path = os.path.join(temp_dir, "temp_img1.png")
                            img2_path = os.path.join(temp_dir, "temp_img2.png")
                            
                            img1.save(img1_path)
                            img2.save(img2_path)
                            
                            # Detect changes
                            change_mask = quick_detect(img1_path, img2_path)
                            
                            # Display results
                            st.image(change_mask, caption="Change Mask", width=400)
                            
                            # Get user ID safely
                            user_data = st.session_state.get('user', {})
                            user_id = user_data.get('uid')
                            
                            if user_id:
                                if save_detection_session(user_id, img1_path, img2_path, change_mask):
                                    st.success("Detection results saved successfully!")
                                else:
                                    st.error("Failed to save detection results")
                            else:
                                st.error("User session expired. Please login again.")
                                st.rerun()

        with tab2:
            show_history()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
