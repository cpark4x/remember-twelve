#!/usr/bin/env python3
"""
Try accessing Google Photos through Google Drive API.
Google Photos are sometimes accessible via Drive.
"""

import pickle
from pathlib import Path
from googleapiclient.discovery import build

token_file = Path('token.json')

if not token_file.exists():
    print("❌ No token found. Run test_google_photos_auth.py first")
    exit(1)

with open(token_file, 'rb') as f:
    creds = pickle.load(f)

print("🔍 Trying to access Google Photos via Drive API...")
print()

try:
    # Try Google Drive API to see if we can access photos
    drive_service = build('drive', 'v3', credentials=creds)

    # Search for photos
    results = drive_service.files().list(
        pageSize=10,
        q="mimeType contains 'image/'",
        fields="files(id, name, mimeType, createdTime)"
    ).execute()

    items = results.get('files', [])

    if items:
        print(f"✅ Found {len(items)} images via Drive API!")
        for item in items:
            print(f"   - {item['name']} ({item['mimeType']})")
    else:
        print("⚠️  No images found via Drive API")

except Exception as e:
    print(f"❌ Drive API error: {e}")

print()
print("💡 This was just a test to see if Drive API gives us access")
print("   Even if it worked, Google Photos Library API is the proper way")
