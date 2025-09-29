# supabase_utils.py

import streamlit as st
from supabase import create_client, Client
from datetime import date, time

def init_connection():
    """Initializes and returns a Supabase client instance."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return None

supabase: Client = init_connection()

def save_user_details(name: str, birth_date: date, birth_time: time, birth_place: str):
    """
    Saves the user's details to the Supabase 'users' table.
    """
    if supabase is None:
        st.error("Database connection is not available.")
        return

    try:
        user_data = {
            "name": name,
            "birth_date": str(birth_date),
            "birth_time": str(birth_time),
            "birth_place": birth_place,
        }
        
        # Insert data into the 'users' table
        response = supabase.table('users').insert(user_data).execute()

        # You can add more robust error checking based on the response if needed
        print("Supabase response:", response)

    except Exception as e:
        # Show an error message in the app if saving fails
        st.error(f"An error occurred while saving your details: {e}")