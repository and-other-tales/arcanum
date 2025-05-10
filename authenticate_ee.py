#!/usr/bin/env python3
import os
import ee

# Path to service account key file
SERVICE_ACCOUNT_KEY = os.path.expanduser('~/arcanum/key.json')

# Initialize Earth Engine with service account
def initialize_ee():
    credentials = ee.ServiceAccountCredentials(
        email=None,  # Will be read from the key file
        key_file=SERVICE_ACCOUNT_KEY
    )
    
    try:
        ee.Initialize(credentials)
        print("Google Earth Engine initialized successfully with service account.")
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return False

if __name__ == "__main__":
    initialize_ee()