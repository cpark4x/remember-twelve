#!/usr/bin/env python3
import pickle
import requests

with open('token.json', 'rb') as f:
    creds = pickle.load(f)

print(f"Testing with token: {creds.token[:30]}...")
print(f"Scope: {creds.scopes}")
print()

# Test the API
response = requests.get(
    'https://photoslibrary.googleapis.com/v1/mediaItems',
    headers={'Authorization': f'Bearer {creds.token}'},
    params={'pageSize': 5}
)

print(f"Status: {response.status_code}")
print(f"Response: {response.text[:500]}")
