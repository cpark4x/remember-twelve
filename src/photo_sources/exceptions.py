"""
Photo Sources Exception Classes

Defines all exceptions that can be raised by photo sources.
"""


class PhotoSourceError(Exception):
    """Base exception for photo source errors."""
    pass


class AuthenticationError(PhotoSourceError):
    """
    Raised when OAuth authentication fails.

    Causes:
    - User denied consent
    - Invalid credentials
    - Token expired and refresh failed
    - Network error during auth flow
    """
    pass


class RateLimitError(PhotoSourceError):
    """
    Raised when API rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying
    """
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(PhotoSourceError):
    """
    Raised when network connection fails.

    Causes:
    - No internet connection
    - API endpoint unreachable
    - DNS resolution failure
    - Connection timeout
    """
    pass


class DownloadError(PhotoSourceError):
    """
    Raised when photo download fails.

    Attributes:
        photo_id: ID of the photo that failed
        reason: Human-readable reason for failure
    """
    def __init__(self, photo_id: str, reason: str):
        super().__init__(f"Failed to download photo {photo_id}: {reason}")
        self.photo_id = photo_id
        self.reason = reason


class InsufficientSpaceError(PhotoSourceError):
    """
    Raised when insufficient disk space for cache.

    Attributes:
        required_mb: Space required in MB
        available_mb: Space available in MB
    """
    def __init__(self, required_mb: int, available_mb: int):
        super().__init__(
            f"Insufficient disk space: need {required_mb}MB, have {available_mb}MB"
        )
        self.required_mb = required_mb
        self.available_mb = available_mb


class InvalidCredentialsError(PhotoSourceError):
    """
    Raised when OAuth credentials are invalid.

    Causes:
    - credentials.json missing
    - credentials.json malformed
    - Wrong credential type (web vs desktop)
    - Credentials revoked
    """
    pass
