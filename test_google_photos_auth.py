#!/usr/bin/env python3
"""
Test Google Photos API authentication.
This script verifies that OAuth credentials are working.
"""

import os
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Scopes required for Google Photos API (read-only)
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

def authenticate_google_photos():
    """
    Authenticate with Google Photos API using OAuth 2.0.
    Returns authenticated service object.
    """
    creds = None
    token_file = Path('token.json')
    credentials_file = Path('google_photos_credentials.json')

    # Check if credentials file exists
    if not credentials_file.exists():
        print("❌ Error: google_photos_credentials.json not found!")
        print(f"   Please place your OAuth credentials at: {credentials_file.absolute()}")
        return None

    # Load token if it exists (from previous authentication)
    if token_file.exists():
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("🔐 Starting OAuth authentication flow...")
            print("   Your browser will open for Google sign-in")
            print()
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_file), SCOPES)
            # Try with explicit port and better error handling
            try:
                creds = flow.run_local_server(port=8080, prompt='consent', open_browser=True)
            except OSError:
                # If port 8080 fails, try a random port
                creds = flow.run_local_server(port=0, prompt='consent', open_browser=True)

        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
        print("✅ Token saved for future use")

    return creds

def test_google_photos_connection(creds):
    """Test connection by fetching basic library info."""
    try:
        import requests

        print("\n📸 Testing Google Photos API connection...")
        print("=" * 60)

        # Use requests directly with proper headers
        headers = {
            'Authorization': f'Bearer {creds.token}',
            'Content-Type': 'application/json'
        }

        response = requests.get(
            'https://photoslibrary.googleapis.com/v1/mediaItems',
            headers=headers,
            params={'pageSize': 10}
        )

        if response.status_code != 200:
            print(f"❌ API Error ({response.status_code}): {response.text}")
            return False

        data = response.json()
        items = data.get('mediaItems', [])

        if not items:
            print("⚠️  No photos found in your Google Photos library")
            print("   Make sure you have photos uploaded to Google Photos")
        else:
            print(f"✅ Successfully connected to Google Photos!")
            print(f"   Found {len(items)} photos (showing first 10)")
            print()
            print("Sample photos:")
            for i, item in enumerate(items[:5], 1):
                filename = item.get('filename', 'Unknown')
                creation_time = item.get('mediaMetadata', {}).get('creationTime', 'Unknown')
                print(f"   {i}. {filename} (created: {creation_time})")

        print("=" * 60)
        print("✅ Authentication test successful!")
        print()
        return True

    except Exception as e:
        print(f"❌ Error testing Google Photos API: {e}")
        return False

def main():
    print("🎨 Remember Twelve - Google Photos Authentication Test")
    print("=" * 60)
    print()

    # Authenticate
    creds = authenticate_google_photos()

    if not creds:
        print("\n❌ Authentication failed")
        return

    print("\n✅ Authentication successful!")

    # Test connection
    test_google_photos_connection(creds)

    print("\n💡 Next steps:")
    print("   1. Authentication is working!")
    print("   2. token.json has been saved (don't commit this to git)")
    print("   3. Ready to implement full Google Photos integration")
    print()

if __name__ == '__main__':
    main()
