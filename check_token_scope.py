#!/usr/bin/env python3
"""Check what scopes are in the saved token."""

import pickle
from pathlib import Path

token_file = Path('token.json')

if not token_file.exists():
    print("❌ No token.json found")
else:
    with open(token_file, 'rb') as f:
        creds = pickle.load(f)

    print("🔍 Token Information:")
    print(f"   Valid: {creds.valid}")
    print(f"   Expired: {creds.expired}")
    print(f"   Scopes: {creds.scopes}")
    print()

    if 'https://www.googleapis.com/auth/photoslibrary.readonly' in creds.scopes:
        print("✅ Correct scope is present!")
    else:
        print("❌ Missing photoslibrary.readonly scope!")
        print("   Need to re-authenticate with correct scope")
