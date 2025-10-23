# Google Photos Integration - Technical Architecture

**Feature:** Feature 1.5 - Google Photos Integration
**Epic:** Epic 1 - AI-Powered Photo Curation Engine
**Architect:** Zen Architect (Ruthless Simplicity)
**Date:** 2025-10-14
**Status:** Architecture Design Complete

---

## Executive Summary

Direct integration with Google Photos API enabling seamless photo curation from cloud storage. Users authenticate via OAuth 2.0, system fetches and temporarily caches photos for analysis, integrates with existing quality and emotional analyzers, and returns curated results with links back to Google Photos.

**Key Principle:** Build cloud photo access as a simple abstraction layer that transparently integrates with existing analyzers. No changes to analyzer code required.

---

## Architecture Philosophy

Following **Ruthless Simplicity**:

1. **PhotoSource Abstraction** - Single interface for local and cloud photos
2. **Transparent Integration** - Existing analyzers work unchanged
3. **Temporary Storage** - No permanent photo storage, privacy-first
4. **Explicit Flow** - Clear data flow from OAuth → Fetch → Cache → Analyze → Cleanup
5. **Modular Components** - Each component has one clear responsibility

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│           (CLI / Web - "Connect Google Photos")              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   PhotoSourceFactory                         │
│         (Creates: LocalPhotoSource | GooglePhotosSource)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
    ┌─────────────────────┐  ┌──────────────────────┐
    │ LocalPhotoSource    │  │ GooglePhotosSource   │
    │  (existing)         │  │      (new)           │
    └─────────────────────┘  └──────────┬───────────┘
                                        │
                            ┌───────────┴────────────┐
                            ▼                        ▼
                ┌────────────────────┐  ┌────────────────────┐
                │ GooglePhotosClient │  │   TokenManager     │
                │  (API Wrapper)     │  │ (OAuth Storage)    │
                └─────────┬──────────┘  └────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Google Photos API   │
              │  (photoslibrary.*)    │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   TempPhotoCache      │
              │  (downloads photos)   │
              └───────────┬───────────┘
                          │
                          ▼
          ┌───────────────────────────────────┐
          │   Existing Analysis Pipeline      │
          │  (PhotoQualityAnalyzer,           │
          │   EmotionalSignificanceDetector)  │
          └───────────────┬───────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ TwelveCurator  │
                 └────────────────┘
```

---

## Core Components

### 1. PhotoSource Interface (Abstraction Layer)

**Purpose:** Define common interface for all photo sources (local, Google Photos, future: iCloud, etc.)

**Location:** `src/photo_sources/base.py`

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List
from pathlib import Path
from datetime import datetime

class PhotoSource(ABC):
    """Abstract base class for photo sources."""

    @abstractmethod
    def scan(
        self,
        year: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Iterator[str]:
        """
        Scan for photos, optionally filtered by date.

        Returns:
            Iterator of photo paths (local temp paths)
        """
        pass

    @abstractmethod
    def get_metadata(self, photo_path: str) -> dict:
        """
        Get metadata for a photo.

        Returns:
            Dict with keys: timestamp, location, format, etc.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up temporary resources (caches, etc.)"""
        pass

    @abstractmethod
    def get_original_url(self, photo_path: str) -> Optional[str]:
        """
        Get original source URL (for Google Photos links).

        Returns:
            URL string or None for local photos
        """
        pass
```

**Contract:**
- Input: Date filters (year, date range)
- Output: Iterator of local file paths (real or temp)
- Side Effects: May download/cache photos temporarily
- Guarantees: All returned paths are valid, readable image files

---

### 2. GooglePhotosSource (Cloud Implementation)

**Purpose:** Implement PhotoSource for Google Photos, handling OAuth, fetching, and caching

**Location:** `src/photo_sources/google_photos_source.py`

**Key Responsibilities:**
1. Implement PhotoSource interface
2. Use GooglePhotosClient for API calls
3. Download photos to temp cache
4. Return temp file paths to analyzers
5. Track mapping: temp_path → google_photos_id → original_url
6. Clean up temp cache on completion

**Dependencies:**
- `GooglePhotosClient` - API communication
- `TempPhotoCache` - Temporary storage
- `TokenManager` - OAuth credential management

**Key Methods:**

```python
class GooglePhotosSource(PhotoSource):
    def __init__(self, credentials_path: str, cache_dir: Optional[Path] = None):
        self.client = GooglePhotosClient(credentials_path)
        self.cache = TempPhotoCache(cache_dir)
        self.photo_map = {}  # temp_path → google_photos_metadata

    def scan(self, year: int, **kwargs) -> Iterator[str]:
        """
        1. Authenticate if needed
        2. Fetch photo list from Google Photos (by year)
        3. Download each photo to temp cache
        4. Yield temp file path
        5. Store mapping for later URL retrieval
        """
        pass

    def get_original_url(self, photo_path: str) -> Optional[str]:
        """Return Google Photos URL from mapping"""
        metadata = self.photo_map.get(photo_path)
        return metadata.get('url') if metadata else None
```

**Error Handling:**
- OAuth failures → Clear error message, re-authentication prompt
- API rate limits → Exponential backoff, resume capability
- Download failures → Skip photo, log error, continue
- Network timeouts → Retry with backoff

---

### 3. GooglePhotosClient (API Wrapper)

**Purpose:** Clean, focused wrapper for Google Photos API interactions

**Location:** `src/photo_sources/google_photos_client.py`

**Key Responsibilities:**
1. OAuth 2.0 authentication flow
2. Token refresh handling
3. API request/response handling
4. Rate limiting and retry logic
5. Error translation to domain errors

**Contract:**
- Input: OAuth credentials, API request parameters
- Output: Structured photo metadata, download URLs
- Side Effects: Token refresh, API calls
- Errors: Raises specific exceptions (AuthError, RateLimitError, APIError)

**Key Methods:**

```python
class GooglePhotosClient:
    def __init__(self, credentials_path: str, token_manager: TokenManager):
        self.credentials_path = credentials_path
        self.token_manager = token_manager
        self.service = None  # Google API service object

    def authenticate(self) -> None:
        """
        Perform OAuth 2.0 flow:
        1. Check if valid token exists
        2. If not, start OAuth flow (browser-based)
        3. Store tokens via TokenManager
        """
        pass

    def list_photos(
        self,
        start_date: datetime,
        end_date: datetime,
        page_size: int = 100
    ) -> Iterator[dict]:
        """
        List photos in date range with pagination.
        Uses: photoslibrary.mediaItems.search

        Returns:
            Iterator of photo metadata dicts
        """
        pass

    def get_download_url(self, photo_id: str) -> str:
        """
        Get temporary download URL for photo.
        URL format: baseUrl + "=d" (for download)
        """
        pass

    def download_photo(self, photo_id: str, output_path: Path) -> None:
        """Download photo to local file with retry logic"""
        pass
```

**API Endpoints Used:**
- `photoslibrary.mediaItems.search` - Search photos by date
- `photoslibrary.mediaItems.get` - Get photo details
- `baseUrl` + `=d` - Download photo content

**Rate Limiting Strategy:**
- Google Photos API limit: 10,000 requests/day
- Implement exponential backoff: 1s → 2s → 4s → 8s
- Batch requests where possible
- Track daily usage, warn at 80% threshold

---

### 4. TokenManager (OAuth Security)

**Purpose:** Secure storage and management of OAuth tokens

**Location:** `src/photo_sources/token_manager.py`

**Key Responsibilities:**
1. Encrypt tokens at rest
2. Automatic token refresh
3. Secure token revocation
4. Multi-user token management (future)

**Security Design:**

```python
class TokenManager:
    """
    Manages OAuth tokens with encryption.

    Storage format: SQLite with encrypted token fields
    Encryption: Fernet (symmetric encryption via cryptography library)
    Key storage: OS keyring (keyring library)
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)

    def store_token(self, user_email: str, token_data: dict) -> None:
        """
        Store encrypted token:
        1. Serialize token_data to JSON
        2. Encrypt with Fernet
        3. Store in SQLite with user_email as key
        """
        pass

    def get_token(self, user_email: str) -> Optional[dict]:
        """Retrieve and decrypt token"""
        pass

    def refresh_token(self, user_email: str) -> dict:
        """
        Refresh expired token:
        1. Use refresh_token from stored data
        2. Call OAuth refresh endpoint
        3. Store new token
        4. Return fresh token
        """
        pass

    def revoke_token(self, user_email: str) -> None:
        """
        Revoke token and delete from storage:
        1. Call Google OAuth revoke endpoint
        2. Delete from local storage
        3. Clear encryption key
        """
        pass

    def _get_or_create_key(self) -> bytes:
        """Get encryption key from OS keyring or create new"""
        pass
```

**Security Principles:**
1. **Encryption at rest** - Tokens never stored in plaintext
2. **OS keyring** - Encryption keys stored in system keyring
3. **Minimal scope** - Only request `photoslibrary.readonly`
4. **Automatic cleanup** - Delete tokens on revocation
5. **No logging** - Never log token contents

---

### 5. TempPhotoCache (Temporary Storage)

**Purpose:** Manage temporary photo downloads with automatic cleanup

**Location:** `src/photo_sources/temp_photo_cache.py`

**Key Responsibilities:**
1. Download photos to temporary directory
2. Track photo lifecycle
3. Automatic cleanup on completion
4. Disk space management

**Design:**

```python
class TempPhotoCache:
    """
    Temporary cache for downloaded photos.

    Uses system temp directory by default.
    Cleanup: Automatic on exit, manual on demand.
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: int = 1000):
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "remember_twelve_cache"
        self.max_size_mb = max_size_mb
        self.cached_files = []  # Track for cleanup
        self._ensure_cache_dir()

    def download_and_cache(
        self,
        photo_id: str,
        download_url: str,
        filename: str
    ) -> Path:
        """
        Download photo to cache:
        1. Check disk space
        2. Download to temp file
        3. Verify file integrity
        4. Track for cleanup
        5. Return path
        """
        pass

    def cleanup(self, force: bool = False) -> None:
        """
        Clean up cached files:
        - Delete all downloaded photos
        - Remove cache directory
        - Clear tracking list
        """
        pass

    def get_cache_size(self) -> int:
        """Get current cache size in MB"""
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatic cleanup on context exit"""
        self.cleanup()
```

**Cleanup Strategy:**
1. **Automatic** - On context manager exit
2. **Manual** - User-triggered "Clear Cache" button
3. **Size-based** - Auto-cleanup if exceeds max_size_mb
4. **Session-based** - Clear on application exit

---

### 6. LocalPhotoSource (Existing Implementation)

**Purpose:** Wrap existing LibraryScanner as PhotoSource implementation

**Location:** `src/photo_sources/local_photo_source.py`

**Implementation:**

```python
class LocalPhotoSource(PhotoSource):
    """
    Adapter for local file system photos.
    Wraps existing LibraryScanner.
    """

    def __init__(self, root_path: str):
        self.scanner = LibraryScanner()
        self.root_path = root_path

    def scan(self, year: Optional[int] = None, **kwargs) -> Iterator[str]:
        """
        Scan local directory, optionally filter by year.
        Uses existing LibraryScanner.
        """
        for photo_path in self.scanner.scan(self.root_path):
            # Filter by year if needed (check EXIF)
            if year and not self._matches_year(photo_path, year):
                continue
            yield photo_path

    def get_metadata(self, photo_path: str) -> dict:
        """Extract EXIF metadata"""
        # Use existing exif_utils from twelve_curator
        pass

    def cleanup(self) -> None:
        """No cleanup needed for local files"""
        pass

    def get_original_url(self, photo_path: str) -> Optional[str]:
        """Local photos have no URL"""
        return None
```

---

### 7. PhotoSourceFactory (Source Selection)

**Purpose:** Create appropriate PhotoSource based on configuration

**Location:** `src/photo_sources/factory.py`

```python
class PhotoSourceFactory:
    """
    Factory for creating photo sources.
    Determines source type from configuration.
    """

    @staticmethod
    def create(config: dict) -> PhotoSource:
        """
        Create appropriate PhotoSource:

        config = {
            'source_type': 'local' | 'google_photos',
            'local_path': '/path/to/photos',  # if local
            'credentials_path': '/path/to/creds.json'  # if google_photos
        }
        """
        source_type = config.get('source_type', 'local')

        if source_type == 'local':
            return LocalPhotoSource(config['local_path'])

        elif source_type == 'google_photos':
            return GooglePhotosSource(config['credentials_path'])

        else:
            raise ValueError(f"Unknown source type: {source_type}")
```

---

## Authentication Flow

### OAuth 2.0 Implementation

**Scope Required:** `https://www.googleapis.com/auth/photoslibrary.readonly`

**Flow:**

```
1. User clicks "Connect Google Photos"
   ↓
2. Check if token exists (TokenManager)
   ↓
   Yes → Validate token → Use it
   ↓
   No/Expired → Start OAuth flow
   ↓
3. OAuth Flow:
   a. Generate authorization URL with scope
   b. Open browser to consent screen
   c. User approves access
   d. Callback with authorization code
   e. Exchange code for token
   ↓
4. Store encrypted token (TokenManager)
   ↓
5. Confirm connection: "Connected: user@gmail.com"
```

**Token Refresh:**
- Tokens expire after 1 hour
- Refresh tokens valid for 6 months
- Auto-refresh before API calls if expired
- Re-authenticate if refresh token expired

**Error Scenarios:**
- **User denies consent** → Clear message, option to retry
- **Network failure** → Retry with exponential backoff
- **Invalid credentials** → Check credentials.json format
- **Token revoked** → Force re-authentication

---

## Data Flow Architecture

### End-to-End Curation Flow

```
1. User Input:
   - Select "Google Photos" as source
   - Choose year (e.g., 2024)
   - Click "Start Curation"

2. Source Creation (PhotoSourceFactory):
   - Create GooglePhotosSource
   - Authenticate if needed

3. Photo Scanning (GooglePhotosSource.scan):
   - Fetch photo list from Google Photos API
     Query: photos from 2024-01-01 to 2024-12-31
   - Paginate through results (100 per page)
   - For each photo:
     a. Download to TempPhotoCache
     b. Store mapping: temp_path → photo_id → URL
     c. Yield temp_path

4. Analysis (Existing Analyzers):
   - PhotoQualityAnalyzer.analyze_photo(temp_path)
     → Quality score

   - EmotionalSignificanceDetector.analyze(temp_path)
     → Emotional score

   - Combine scores per CurationConfig
     → Combined score

5. Curation (TwelveCurator):
   - Apply diversity filter
   - Apply monthly distribution
   - Select top 12 photos
   - Create TwelveSelection

6. Results Enhancement:
   - For each selected photo:
     a. Get original URL from GooglePhotosSource
     b. Add to PhotoCandidate.metadata['google_photos_url']

7. Cleanup:
   - TempPhotoCache.cleanup()
   - Delete all downloaded photos
   - Preserve metadata and results

8. Output:
   - Display 12 selected photos
   - Each photo: thumbnail + link to Google Photos
   - Save results JSON with URLs
```

---

## Integration with Existing Components

### Zero Changes to Analyzers

**Key Design Principle:** Analyzers work with file paths. They don't care if it's local or downloaded from cloud.

**PhotoQualityAnalyzer Integration:**
```python
# analyzer.py - NO CHANGES NEEDED
def analyze_photo(self, photo_path: str) -> QualityScore:
    """
    Analyze photo quality.

    photo_path can be:
    - Local file: /Users/john/Photos/IMG_1234.jpg
    - Temp cached: /tmp/remember_twelve_cache/photo_abc123.jpg

    Analyzer doesn't care which - just reads the file.
    """
    image = self._load_image(photo_path)
    sharpness = self.sharpness_detector.analyze(image)
    exposure = self.exposure_detector.analyze(image)
    return self._compute_score(sharpness, exposure)
```

**EmotionalSignificanceDetector Integration:**
```python
# emotional_significance/detector.py - NO CHANGES NEEDED
def analyze(self, photo_path: str) -> EmotionalScore:
    """
    Detect emotional significance.
    Works with any file path.
    """
    faces = self.face_detector.detect(photo_path)
    emotions = self.emotion_analyzer.analyze(photo_path)
    return self._compute_score(faces, emotions)
```

**TwelveCurator Integration:**
```python
# twelve_curator/curator.py - MINIMAL CHANGES
class TwelveCurator:
    def curate_year(
        self,
        photo_source: PhotoSource,  # NEW: Accept PhotoSource
        year: int,
        config: CurationConfig
    ) -> TwelveSelection:
        """
        Curate 12 photos from source.

        OLD: curate_year(photo_directory: str, ...)
        NEW: curate_year(photo_source: PhotoSource, ...)
        """
        candidates = []

        # Scan photos from source (local or cloud)
        for photo_path in photo_source.scan(year=year):
            # Analyze quality
            quality_score = self.quality_analyzer.analyze_photo(photo_path)

            # Analyze emotional significance
            emotional_score = self.emotional_analyzer.analyze(photo_path)

            # Get metadata (timestamp, location)
            metadata = photo_source.get_metadata(photo_path)

            # Create candidate
            candidate = PhotoCandidate(
                photo_path=Path(photo_path),
                timestamp=metadata.get('timestamp'),
                month=metadata.get('month'),
                quality_score=quality_score.composite,
                emotional_score=emotional_score.score,
                combined_score=self._combine_scores(quality_score, emotional_score, config),
                metadata={
                    'original_url': photo_source.get_original_url(photo_path),
                    **metadata
                }
            )
            candidates.append(candidate)

        # Rest of curation logic unchanged
        selected = self._select_twelve(candidates, config)

        # Cleanup temporary cache
        photo_source.cleanup()

        return TwelveSelection(...)
```

**Change Summary:**
- PhotoQualityAnalyzer: **0 changes**
- EmotionalSignificanceDetector: **0 changes**
- TwelveCurator: **~10 lines changed** (accept PhotoSource instead of directory path)

---

## Error Handling Strategy

### Error Categories and Responses

**1. Authentication Errors**
- **Scenario:** OAuth failure, token expired, credentials invalid
- **Response:**
  - Clear error message to user
  - "Re-authenticate" button
  - Do not retry automatically (user action required)

**2. Network Errors**
- **Scenario:** Connection timeout, DNS failure, API unreachable
- **Response:**
  - Retry with exponential backoff (3 attempts)
  - Show "Network issue, retrying..." message
  - Fail gracefully after 3 attempts

**3. Rate Limit Errors**
- **Scenario:** Hit Google Photos API daily/per-minute limit
- **Response:**
  - Pause requests
  - Exponential backoff: 1s → 2s → 4s → 8s
  - Show "Rate limited, slowing down..." message
  - Resume automatically

**4. Download Errors**
- **Scenario:** Single photo download fails
- **Response:**
  - Log error with photo_id
  - Skip photo, continue with others
  - Show "X photos failed to download" summary
  - Do not fail entire curation

**5. Disk Space Errors**
- **Scenario:** Insufficient space for photo cache
- **Response:**
  - Check disk space before starting
  - Estimate required space (photo_count * avg_photo_size)
  - Fail fast with clear message
  - Suggest cache cleanup or more space

**6. API Errors**
- **Scenario:** Unexpected Google Photos API error
- **Response:**
  - Log full error details
  - Show user-friendly message
  - Offer "Report Issue" option
  - Do not expose technical details to user

### Error Response Classes

```python
class GooglePhotosError(Exception):
    """Base exception for Google Photos integration"""
    pass

class AuthenticationError(GooglePhotosError):
    """OAuth authentication failed"""
    pass

class RateLimitError(GooglePhotosError):
    """API rate limit exceeded"""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

class NetworkError(GooglePhotosError):
    """Network connection error"""
    pass

class DownloadError(GooglePhotosError):
    """Photo download failed"""
    def __init__(self, photo_id: str, reason: str):
        self.photo_id = photo_id
        self.reason = reason

class InsufficientSpaceError(GooglePhotosError):
    """Not enough disk space for cache"""
    def __init__(self, required_mb: int, available_mb: int):
        self.required_mb = required_mb
        self.available_mb = available_mb
```

---

## Security Considerations

### OAuth Token Security

**Storage:**
- Tokens encrypted with Fernet (symmetric encryption)
- Encryption key stored in OS keyring (never in code/config)
- SQLite database for token storage (not plain files)
- Database file permissions: 600 (owner read/write only)

**Access Control:**
- Minimal scope: `photoslibrary.readonly` only
- No write access to Google Photos
- No access to other Google services
- User can revoke at any time via Google Account settings

**Token Lifecycle:**
- Access token: 1 hour expiration (auto-refresh)
- Refresh token: 6 months expiration (re-auth required)
- Automatic cleanup on app uninstall
- Manual revocation via "Disconnect" button

### Photo Data Privacy

**Principles:**
1. **Zero Permanent Storage** - Photos deleted after curation
2. **Explicit Consent** - Clear privacy message before connection
3. **Local Processing** - Analysis happens on user's machine
4. **No Transmission** - Photos never sent to external servers
5. **Cache Transparency** - User can view/clear cache at any time

**Privacy Guarantees:**
- Photos downloaded only during active curation
- Automatic cleanup after curation completes
- Manual "Clear Cache" button always available
- No analytics or tracking of photo content
- No logging of photo filenames or metadata

### Credential Security

**credentials.json (OAuth Client Secret):**
- Not committed to git (.gitignore)
- User must obtain from Google Cloud Console
- Instructions in README for setup
- No hardcoded credentials in code

**Best Practices:**
- Never log token contents
- Never log photo URLs (contain auth tokens)
- Sanitize errors before showing to user
- Clear tokens from memory after use

---

## Rate Limiting and Performance

### Google Photos API Limits

**Official Limits:**
- 10,000 requests per day per project
- Burst limit: ~1 request per second sustained

**Our Strategy:**
- **Batch requests** where possible (100 photos per search request)
- **Request estimation**: Calculate expected requests before starting
  - Example: 1000 photos → ~10 search requests + 1000 download requests = 1010 requests
- **Daily tracking**: Store request count, reset at midnight
- **Warning threshold**: Alert user at 8000 requests (80% of limit)
- **Graceful degradation**: Pause if approaching limit

### Performance Optimization

**Parallel Downloads:**
```python
# Download photos in parallel (max 5 concurrent)
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
        executor.submit(download_photo, photo)
        for photo in photo_batch
    ]
    for future in as_completed(futures):
        photo_path = future.result()
        yield photo_path
```

**Progressive Analysis:**
- Don't wait for all downloads to complete
- Analyze photos as they arrive
- Show real-time progress: "Analyzed 45/312 photos..."

**Caching Strategy:**
- Download only photos needed for analysis
- If re-running curation, check if photos already cached
- Skip re-download if cache valid (session-based)

**Expected Performance:**
- 100 photos: ~2-3 minutes (download + analyze)
- 500 photos: ~10-12 minutes
- 1000 photos: ~20-25 minutes

**Progress Indicators:**
```
Connecting to Google Photos... ✓
Fetching photo list... 312 photos found
Downloading photos... [████████░░░░░░░░] 45/312 (14%)
Analyzing quality... [██████░░░░░░░░░░] 38/312 (12%)
```

---

## Implementation Modules

### Module Breakdown

```
src/photo_sources/
├── __init__.py                 # Public API exports
├── base.py                     # PhotoSource interface
├── factory.py                  # PhotoSourceFactory
├── local_photo_source.py       # LocalPhotoSource (adapter)
├── google_photos_source.py     # GooglePhotosSource
├── google_photos_client.py     # API wrapper
├── token_manager.py            # OAuth token management
└── temp_photo_cache.py         # Temporary storage

Dependencies (new):
- google-auth
- google-auth-oauthlib
- google-auth-httplib2
- google-api-python-client
- cryptography (for Fernet encryption)
- keyring (for key storage)
```

### Dependency Graph

```
TwelveCurator
    └── PhotoSource (interface)
            ├── LocalPhotoSource
            │       └── LibraryScanner (existing)
            │
            └── GooglePhotosSource
                    ├── GooglePhotosClient
                    │       └── TokenManager
                    │
                    └── TempPhotoCache
```

---

## API Contract Specifications

### PhotoSource Interface Contract

**Method: scan()**
```python
def scan(
    year: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Iterator[str]:
    """
    Scan for photos with optional date filtering.

    Args:
        year: Filter photos to this year (2020-2024)
        start_date: Filter photos after this date
        end_date: Filter photos before this date

    Returns:
        Iterator yielding absolute file paths (local or temp)

    Yields:
        str: Absolute path to photo file

    Raises:
        AuthenticationError: If OAuth fails (Google Photos)
        NetworkError: If connection fails
        ValueError: If invalid date range

    Guarantees:
        - All yielded paths are valid, readable files
        - Files are images (jpg, png, heic, heif)
        - Files exist at time of yield
        - Paths are absolute (not relative)
    """
```

**Method: get_metadata()**
```python
def get_metadata(photo_path: str) -> dict:
    """
    Get metadata for a photo.

    Args:
        photo_path: Path returned by scan()

    Returns:
        Dict with keys:
            - timestamp: datetime | None
            - month: int (1-12) | None
            - location: dict (lat, lon) | None
            - format: str (jpg, png, heic)
            - width: int | None
            - height: int | None
            - file_size: int (bytes)

    Raises:
        FileNotFoundError: If photo_path doesn't exist
        ValueError: If photo_path not from this source
    """
```

### GooglePhotosClient API Contract

**Method: list_photos()**
```python
def list_photos(
    start_date: datetime,
    end_date: datetime,
    page_size: int = 100
) -> Iterator[dict]:
    """
    List photos in date range.

    Args:
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        page_size: Results per page (max 100)

    Returns:
        Iterator of photo metadata dicts

    Yields:
        dict: {
            'id': str,
            'filename': str,
            'baseUrl': str (temp URL, valid 60 mins),
            'mediaMetadata': {
                'creationTime': str (ISO 8601),
                'width': str,
                'height': str
            }
        }

    Raises:
        AuthenticationError: If not authenticated
        RateLimitError: If rate limit exceeded
        NetworkError: If API unreachable
    """
```

---

## Testing Strategy

### Unit Tests

**1. PhotoSource Implementations**
```python
# test_local_photo_source.py
def test_scan_returns_valid_paths():
    source = LocalPhotoSource("/test/photos")
    paths = list(source.scan())
    assert all(Path(p).exists() for p in paths)

# test_google_photos_source.py
def test_scan_downloads_to_cache():
    source = GooglePhotosSource(mock_credentials)
    paths = list(source.scan(year=2024))
    assert all(str(p).startswith(str(cache_dir)) for p in paths)
```

**2. TokenManager**
```python
def test_token_encryption():
    manager = TokenManager(test_db)
    token = {'access_token': 'secret123'}
    manager.store_token('user@test.com', token)

    # Verify encrypted in DB
    raw_db_value = read_db_raw(test_db)
    assert 'secret123' not in raw_db_value

def test_token_refresh():
    manager = TokenManager(test_db)
    manager.store_token('user@test.com', expired_token)

    fresh_token = manager.refresh_token('user@test.com')
    assert fresh_token['access_token'] != expired_token['access_token']
```

**3. TempPhotoCache**
```python
def test_cache_cleanup():
    cache = TempPhotoCache(test_dir)
    photo_path = cache.download_and_cache('photo123', 'url', 'test.jpg')

    assert photo_path.exists()
    cache.cleanup()
    assert not photo_path.exists()
    assert not test_dir.exists()
```

### Integration Tests

**1. End-to-End Curation**
```python
def test_google_photos_curation_flow():
    """Test full flow with mock Google Photos API"""
    source = GooglePhotosSource(mock_credentials)
    curator = TwelveCurator(quality_analyzer, emotional_analyzer)

    result = curator.curate_year(source, year=2024)

    assert len(result.photos) == 12
    assert all(p.metadata.get('google_photos_url') for p in result.photos)

    # Verify cleanup
    assert not list(cache_dir.iterdir())
```

**2. OAuth Flow**
```python
def test_oauth_authentication():
    """Test OAuth flow with mock server"""
    client = GooglePhotosClient(credentials_path)

    # Simulate OAuth callback
    with mock_oauth_server():
        client.authenticate()

    assert client.is_authenticated()
    assert TokenManager().get_token('user@test.com')
```

### Mock Data

**Mock Google Photos API responses:**
```python
MOCK_PHOTO_LIST_RESPONSE = {
    'mediaItems': [
        {
            'id': 'photo_123',
            'filename': 'IMG_1234.jpg',
            'baseUrl': 'https://mock-url.com/photo123',
            'mediaMetadata': {
                'creationTime': '2024-06-15T14:30:00Z',
                'width': '4032',
                'height': '3024'
            }
        },
        # ... more photos
    ],
    'nextPageToken': 'token_abc'
}
```

### Test Coverage Goals

- Unit tests: **80%+ coverage**
- Integration tests: **Critical paths covered**
- Error scenarios: **All error types tested**
- Security: **Token encryption/cleanup verified**

---

## Deployment and Configuration

### Environment Configuration

**config.yaml:**
```yaml
photo_sources:
  google_photos:
    enabled: true
    credentials_path: ~/.remember_twelve/google_credentials.json
    cache_dir: ~/.remember_twelve/cache
    max_cache_size_mb: 1000

  local:
    enabled: true
    default_path: ~/Photos

security:
  token_storage: ~/.remember_twelve/tokens.db
  encryption_key_provider: keyring

api:
  rate_limit_buffer: 0.8  # Use 80% of daily limit
  request_timeout_seconds: 30
  max_retries: 3
```

### User Setup Steps

**1. Google Cloud Project Setup:**
```bash
# User must do these steps once:
1. Go to Google Cloud Console
2. Create new project "Remember Twelve"
3. Enable Google Photos Library API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials.json
6. Move to ~/.remember_twelve/google_credentials.json
```

**2. First-Time Connection:**
```bash
# In Remember Twelve app:
1. Click "Connect Google Photos"
2. Browser opens to Google consent screen
3. User approves access (readonly scope)
4. App shows "Connected: user@gmail.com"
5. Ready to curate!
```

**3. Privacy Settings:**
```bash
# User controls:
- "View Cache" → See downloaded photos
- "Clear Cache" → Delete all cached photos
- "Disconnect" → Revoke access, delete tokens
```

---

## Future Enhancements

### Phase 4+ Features (Post-MVP)

**1. Multiple Provider Support**
- iCloud Photos integration (similar architecture)
- Dropbox Photos
- Instagram import

**2. Advanced Caching**
- Persistent cache with expiration (24 hours)
- Resume interrupted curations
- Smart pre-caching based on year selection

**3. Collaborative Curation**
- Share Google Photos albums for curation
- Multi-user access to same library
- Family circle curation from shared albums

**4. Export to Google Photos**
- Create album in Google Photos with selected 12
- One-click "Save Twelve to Album"

**5. Performance Optimizations**
- Incremental curation (only new photos)
- Parallel analysis (multi-threading)
- Thumbnail-based preview before full download

---

## Decision Log

### Key Architectural Decisions

**1. PhotoSource Abstraction**
- **Decision:** Create abstract PhotoSource interface
- **Rationale:** Enables multiple providers (local, Google Photos, iCloud) with zero analyzer changes
- **Trade-off:** Small abstraction layer vs. direct integration
- **Outcome:** Clean separation, easy to extend

**2. Temporary Cache vs. Permanent Storage**
- **Decision:** Use temporary cache, delete after curation
- **Rationale:** Privacy-first, user control, no storage bloat
- **Trade-off:** Re-download for re-curation vs. storage
- **Outcome:** Clear privacy story, acceptable for MVP

**3. Token Encryption at Rest**
- **Decision:** Encrypt OAuth tokens with Fernet + OS keyring
- **Rationale:** Security best practice, protect user credentials
- **Trade-off:** Complexity vs. security
- **Outcome:** Strong security, minimal overhead

**4. No Changes to Analyzers**
- **Decision:** Analyzers work with file paths, unchanged
- **Rationale:** Separation of concerns, simplicity
- **Trade-off:** Slightly less efficient (download all) vs. clean interface
- **Outcome:** Analyzers remain simple, testable, reusable

**5. Progressive Download + Analysis**
- **Decision:** Analyze photos as they download (streaming)
- **Rationale:** Better UX (progress), faster perceived time
- **Trade-off:** Slightly more complex coordination
- **Outcome:** Much better user experience

---

## Success Metrics

### Technical Metrics

**Performance:**
- Average time to curate 500 photos: < 12 minutes
- Cache cleanup time: < 5 seconds
- OAuth authentication time: < 30 seconds

**Reliability:**
- Curation success rate: > 95%
- API error recovery rate: > 90%
- Network error recovery: 100% (with retries)

**Security:**
- Token encryption: 100% (never plaintext)
- Credential exposure: 0 incidents
- Cache cleanup rate: 100%

### User Metrics

**Adoption:**
- % users who connect Google Photos: Target 70%+
- Connection success rate: > 95%
- Re-authentication rate: < 5% per month

**Experience:**
- Time to first curation: < 5 minutes (including auth)
- User-reported privacy satisfaction: > 90%
- Error clarity rating: > 80% clear

---

## Summary: Key Architectural Principles

**1. Ruthless Simplicity**
- Single responsibility per component
- Clear interfaces, minimal abstractions
- Direct solutions over clever patterns

**2. Privacy First**
- Temporary storage only
- Explicit user consent
- Transparent cache management

**3. Security by Design**
- Encrypted tokens at rest
- Minimal OAuth scope
- No credential logging

**4. Seamless Integration**
- Zero changes to existing analyzers
- PhotoSource abstraction
- Clean dependency injection

**5. Graceful Degradation**
- Retry on network errors
- Skip failed photos, continue processing
- Clear error messages

---

## Appendix: File Structure

```
src/
├── photo_quality_analyzer/      # Existing (no changes)
│   ├── analyzer.py
│   ├── metrics/
│   └── scanner.py
│
├── emotional_significance/      # Existing (no changes)
│   └── detector.py
│
├── twelve_curator/              # Minimal changes
│   ├── curator.py               # Accept PhotoSource
│   └── data_classes.py
│
└── photo_sources/               # NEW MODULE
    ├── __init__.py
    ├── base.py                  # PhotoSource interface
    ├── factory.py               # PhotoSourceFactory
    ├── local_photo_source.py    # LocalPhotoSource
    ├── google_photos_source.py  # GooglePhotosSource
    ├── google_photos_client.py  # API wrapper
    ├── token_manager.py         # OAuth tokens
    └── temp_photo_cache.py      # Temp storage

tests/
└── photo_sources/
    ├── test_base.py
    ├── test_factory.py
    ├── test_local_source.py
    ├── test_google_photos_source.py
    ├── test_google_photos_client.py
    ├── test_token_manager.py
    └── test_temp_photo_cache.py
```

---

**Architecture Status:** ✅ Complete
**Next Phase:** Implementation (Phase 1 - Authentication & Basic Fetching)
**Estimated Implementation Time:** 6-9 days (across 3 phases)

---

*Designed with Ruthless Simplicity by Zen Architect*
*"The best design is often the simplest" - Amplifier Philosophy*
