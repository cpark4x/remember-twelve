"""
GooglePhotosClient - Google Photos API Wrapper

Clean, focused wrapper for Google Photos API interactions.
Handles OAuth 2.0 authentication, API requests, rate limiting, and error handling.

Phase 1 Features (MVP):
- OAuth 2.0 authentication flow
- Token refresh handling
- List photos by date range
- Get photo download URLs
- Basic rate limiting

API Documentation:
https://developers.google.com/photos/library/guides/overview
"""

import time
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime, timedelta
import requests

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .token_manager import TokenManager
from .exceptions import (
    AuthenticationError,
    RateLimitError,
    NetworkError,
    InvalidCredentialsError
)


class GooglePhotosClient:
    """
    Google Photos API client with OAuth 2.0 authentication.

    Scopes:
    - photoslibrary.readonly: Read-only access to photos

    Rate Limits:
    - 10,000 requests/day per project
    - ~1 request/second sustained burst
    """

    # OAuth scopes - readonly only for security
    SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']

    # API endpoints
    API_SERVICE_NAME = 'photoslibrary'
    API_VERSION = 'v1'

    # Rate limiting
    MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests

    def __init__(
        self,
        credentials_path: str,
        token_manager: Optional[TokenManager] = None
    ):
        """
        Initialize Google Photos client.

        Args:
            credentials_path: Path to OAuth credentials.json
            token_manager: TokenManager instance (creates new if None)

        Raises:
            InvalidCredentialsError: If credentials file invalid
        """
        self.credentials_path = Path(credentials_path)

        if not self.credentials_path.exists():
            raise InvalidCredentialsError(
                f"Credentials file not found: {credentials_path}"
            )

        # Token management
        if token_manager is None:
            # Default token storage location
            token_db = Path.home() / '.remember_twelve' / 'tokens.db'
            token_manager = TokenManager(token_db)
        self.token_manager = token_manager

        # API service (initialized after auth)
        self.service = None
        self.credentials = None
        self.user_email = None

        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0

    def authenticate(self, user_email: Optional[str] = None) -> str:
        """
        Perform OAuth 2.0 authentication.

        Flow:
        1. Check if valid token exists
        2. If not, start OAuth flow (opens browser)
        3. Store encrypted token
        4. Initialize API service

        Args:
            user_email: User email for token lookup (optional)

        Returns:
            Authenticated user's email address

        Raises:
            AuthenticationError: If OAuth flow fails
        """
        creds = None

        # Try to load existing token
        if user_email:
            token_data = self.token_manager.get_token(user_email)
            if token_data and self.token_manager.is_token_valid(user_email):
                creds = Credentials.from_authorized_user_info(token_data, self.SCOPES)

        # Refresh expired token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Update stored token
                self.token_manager.store_token(user_email, {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes,
                    'expires_in': 3600
                })
                self.token_manager.update_last_refreshed(user_email)
            except Exception as e:
                raise AuthenticationError(f"Token refresh failed: {e}")

        # Start new OAuth flow if no valid token
        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path),
                    self.SCOPES
                )
                creds = flow.run_local_server(port=0)

                # Store new token
                token_data = {
                    'token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'token_uri': creds.token_uri,
                    'client_id': creds.client_id,
                    'client_secret': creds.client_secret,
                    'scopes': creds.scopes,
                    'expires_in': 3600
                }

                # Get user email from token info
                user_email = self._get_user_email(creds)
                self.token_manager.store_token(user_email, token_data)

            except Exception as e:
                raise AuthenticationError(f"OAuth flow failed: {e}")

        # Store credentials (we'll use REST API directly, not discovery)
        self.credentials = creds
        self.user_email = user_email or self._get_user_email(creds)
        self.service = True  # Mark as initialized

        return self.user_email

    def _get_user_email(self, creds: Credentials) -> str:
        """
        Extract user email from credentials.

        Args:
            creds: Google OAuth credentials

        Returns:
            User email address
        """
        try:
            # Use OAuth2 userinfo endpoint
            import google.oauth2.id_token
            import google.auth.transport.requests

            request = google.auth.transport.requests.Request()
            id_info = google.oauth2.id_token.verify_oauth2_token(
                creds.id_token,
                request,
                creds.client_id
            )
            return id_info.get('email', 'unknown@unknown.com')
        except:
            # Fallback: try to extract from token
            return 'user@unknown.com'

    def is_authenticated(self) -> bool:
        """Check if client is authenticated and ready."""
        return self.service is not None and self.credentials is not None

    def list_photos(
        self,
        start_date: datetime,
        end_date: datetime,
        page_size: int = 100
    ) -> Iterator[Dict[str, Any]]:
        """
        List photos in date range with pagination.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            page_size: Results per page (max 100)

        Yields:
            Photo metadata dicts with keys:
                - id: str
                - filename: str
                - baseUrl: str (temp download URL, valid 60 mins)
                - mediaMetadata: dict (creationTime, width, height)
                - mimeType: str
                - productUrl: str (link to Google Photos)

        Raises:
            AuthenticationError: If not authenticated
            RateLimitError: If rate limit exceeded
            NetworkError: If API unreachable

        Example:
            >>> client = GooglePhotosClient(creds_path)
            >>> client.authenticate()
            >>> for photo in client.list_photos(start_date, end_date):
            ...     print(photo['filename'], photo['id'])
        """
        if not self.is_authenticated():
            raise AuthenticationError("Not authenticated. Call authenticate() first.")

        # Build search filter
        search_filter = {
            'dateFilter': {
                'ranges': [{
                    'startDate': {
                        'year': start_date.year,
                        'month': start_date.month,
                        'day': start_date.day
                    },
                    'endDate': {
                        'year': end_date.year,
                        'month': end_date.month,
                        'day': end_date.day
                    }
                }]
            }
        }

        # Paginate through results using direct REST API
        page_token = None
        api_url = 'https://photoslibrary.googleapis.com/v1/mediaItems:search'

        while True:
            try:
                # Rate limiting
                self._wait_for_rate_limit()

                # Prepare request body
                request_body = {
                    'pageSize': min(page_size, 100),
                    'filters': search_filter
                }
                if page_token:
                    request_body['pageToken'] = page_token

                # Get access token
                if self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())

                # Make direct REST API call
                headers = {
                    'Authorization': f'Bearer {self.credentials.token}',
                    'Content-Type': 'application/json'
                }

                response = requests.post(
                    api_url,
                    headers=headers,
                    json=request_body,
                    timeout=30
                )

                self.request_count += 1

                # Handle HTTP errors
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError(
                        "Google Photos API rate limit exceeded",
                        retry_after=retry_after
                    )
                elif response.status_code in [401, 403]:
                    raise AuthenticationError(f"Authentication failed: {response.text}")
                elif response.status_code != 200:
                    raise NetworkError(f"API request failed: {response.status_code} - {response.text}")

                # Parse response
                data = response.json()

                # Yield photos
                media_items = data.get('mediaItems', [])
                for item in media_items:
                    yield item

                # Check for next page
                page_token = data.get('nextPageToken')
                if not page_token:
                    break

            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error: {e}")

    def get_download_url(self, photo_id: str, base_url: str) -> str:
        """
        Get download URL for photo.

        Args:
            photo_id: Photo ID from list_photos()
            base_url: Base URL from photo metadata

        Returns:
            Download URL (add =d for download, =w1024 for resize, etc.)

        Note: Google Photos baseUrl expires after 60 minutes
        """
        # baseUrl format: https://lh3.googleusercontent.com/...
        # Add =d parameter for original quality download
        return f"{base_url}=d"

    def download_photo(self, download_url: str, output_path: Path) -> None:
        """
        Download photo to local file.

        Args:
            download_url: URL from get_download_url()
            output_path: Local file path to save

        Raises:
            NetworkError: If download fails
        """
        try:
            self._wait_for_rate_limit()

            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Photo download failed: {e}")

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        time_since_last = now - self.last_request_time

        if time_since_last < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)

        self.last_request_time = time.time()

    def revoke_access(self) -> None:
        """
        Revoke OAuth token and disconnect.

        Calls Google's revoke endpoint and removes stored token.
        """
        if not self.credentials:
            return

        try:
            # Revoke token with Google
            requests.post(
                'https://oauth2.googleapis.com/revoke',
                params={'token': self.credentials.token},
                headers={'content-type': 'application/x-www-form-urlencoded'}
            )
        except:
            pass  # Best effort

        # Remove from local storage
        if self.user_email:
            self.token_manager.revoke_token(self.user_email)

        # Clear client state
        self.service = None
        self.credentials = None
        self.user_email = None

    def get_request_count(self) -> int:
        """Get number of API requests made this session."""
        return self.request_count
