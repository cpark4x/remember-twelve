## Photo Sources Module

Unified interface for accessing photos from local file system and cloud providers (Google Photos, iCloud, etc.).

### Features

**Phase 1 (Complete):**
- PhotoSource abstract interface
- GooglePhotosClient with OAuth 2.0 authentication
- TokenManager with encrypted token storage
- GooglePhotosSource for fetching photos from Google Photos
- PhotoSourceFactory for creating sources from config

**Coming Soon:**
- Phase 2: TempPhotoCache for better caching strategies
- Phase 3: Full integration with TwelveCurator
- LocalPhotoSource for local file system photos

### Quick Start

#### Google Photos Setup

1. **Create Google Cloud Project:**
   ```bash
   # Go to https://console.cloud.google.com
   # Create project "Remember Twelve"
   # Enable Google Photos Library API
   ```

2. **Create OAuth Credentials:**
   ```bash
   # In Google Cloud Console:
   # APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
   # Application type: Desktop app
   # Download as google_photos_credentials.json
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### Basic Usage

```python
from photo_sources import PhotoSourceFactory

# Create Google Photos source
config = {
    'source_type': 'google_photos',
    'credentials_path': 'google_photos_credentials.json'
}
source = PhotoSourceFactory.create(config)

# Authenticate (opens browser)
user_email = source.authenticate()
print(f"Connected as: {user_email}")

# Fetch photos from 2024
for photo_path in source.scan(year=2024):
    # photo_path is local temp file
    metadata = source.get_metadata(photo_path)
    url = source.get_original_url(photo_path)

    print(f"Photo: {photo_path}")
    print(f"Date: {metadata['timestamp']}")
    print(f"Google Photos: {url}")

# Cleanup temp cache
source.cleanup()
```

#### Context Manager

```python
with PhotoSourceFactory.create(config) as source:
    source.authenticate()

    for photo in source.scan(year=2024):
        # Process photos
        pass

    # Cleanup automatic on exit
```

### Architecture

```
PhotoSource (interface)
├── LocalPhotoSource (coming soon)
└── GooglePhotosSource
    ├── GooglePhotosClient
    │   └── TokenManager
    └── TempPhotoCache (Phase 2)
```

### Security

**Token Storage:**
- Encrypted with Fernet (AES 128)
- Encryption key stored in OS keyring
- SQLite database with encrypted fields
- Never logs token contents

**OAuth Scopes:**
- `photoslibrary.readonly` only
- No write access to Google Photos
- User can revoke at any time

**Privacy:**
- Photos downloaded to temp cache only
- Automatic cleanup after use
- No permanent photo storage
- Local processing only (no external servers)

### API Reference

#### PhotoSource

Abstract base class for all photo sources.

**Methods:**
- `scan(year, start_date, end_date)` → Iterator[str]
- `get_metadata(photo_path)` → Dict[str, Any]
- `get_original_url(photo_path)` → Optional[str]
- `cleanup()` → None

#### GooglePhotosSource

Google Photos implementation of PhotoSource.

**Constructor:**
```python
GooglePhotosSource(
    credentials_path: str,
    cache_dir: Optional[str] = None,
    token_manager: Optional[TokenManager] = None
)
```

**Methods:**
- `authenticate(user_email)` → str

#### PhotoSourceFactory

Factory for creating photo sources.

**Methods:**
- `create(config)` → PhotoSource
- `create_local(path)` → PhotoSource
- `create_google_photos(credentials_path, cache_dir)` → PhotoSource

### Demo

Run the Phase 1 demo:

```bash
python demo_google_photos_phase1.py
```

Demonstrates:
1. OAuth authentication
2. Fetching photos from 2023
3. Extracting metadata
4. Getting Google Photos URLs
5. Cleanup

### Error Handling

```python
from photo_sources.exceptions import (
    AuthenticationError,
    RateLimitError,
    NetworkError,
    DownloadError
)

try:
    source.authenticate()
    for photo in source.scan(year=2024):
        process(photo)

except AuthenticationError as e:
    print(f"Auth failed: {e}")
    # User needs to re-authenticate

except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")

except NetworkError as e:
    print(f"Network error: {e}")
    # Retry with backoff
```

### Rate Limiting

Google Photos API limits:
- 10,000 requests/day per project
- ~1 request/second sustained

Client handles:
- Automatic request throttling (100ms between requests)
- Exponential backoff on rate limit errors
- Request counting and statistics

### Testing

```bash
# Run tests (when implemented)
pytest tests/photo_sources/

# With coverage
pytest --cov=src/photo_sources tests/photo_sources/
```

### Roadmap

**Phase 2: Enhanced Caching**
- TempPhotoCache with disk space management
- Parallel downloads
- Resume capability

**Phase 3: Curator Integration**
- Update TwelveCurator to accept PhotoSource
- End-to-end curation from Google Photos
- Results with Google Photos links

**Future:**
- LocalPhotoSource for local directories
- iCloud Photos integration
- Multiple provider support

### Dependencies

```
google-auth>=2.23.0
google-auth-oauthlib>=1.1.0
google-api-python-client>=2.100.0
cryptography>=41.0.0
keyring>=24.0.0
requests>=2.31.0
```

### License

[To be determined]

---

**Status:** Phase 1 Complete ✅
**Next:** Test with real credentials, then implement Phase 2
