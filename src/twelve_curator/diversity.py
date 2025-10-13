"""
Visual diversity detection for photo curation.

Provides functions to detect visually similar photos and ensure
diversity in the final Twelve selection.

Uses histogram comparison for fast, simple visual similarity detection.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def calculate_similarity(photo_a: Path, photo_b: Path, method: str = "histogram") -> float:
    """
    Calculate visual similarity between two photos.

    Uses histogram comparison by default for speed and simplicity.

    Args:
        photo_a: Path to first photo
        photo_b: Path to second photo
        method: Similarity method ("histogram" only for now)

    Returns:
        Similarity score (0.0 = completely different, 1.0 = identical)

    Raises:
        ValueError: If OpenCV is not available
        FileNotFoundError: If photos don't exist

    Examples:
        >>> similarity = calculate_similarity(photo1, photo2)
        >>> if similarity > 0.85:
        ...     print("Photos are very similar (possible duplicates)")
    """
    if not CV2_AVAILABLE:
        raise ValueError("OpenCV (cv2) is required for visual similarity detection")

    if not photo_a.exists():
        raise FileNotFoundError(f"Photo not found: {photo_a}")
    if not photo_b.exists():
        raise FileNotFoundError(f"Photo not found: {photo_b}")

    if method == "histogram":
        return _calculate_histogram_similarity(photo_a, photo_b)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def is_diverse(
    photo_a: Path,
    photo_b: Path,
    threshold: float = 0.85,
    method: str = "histogram"
) -> bool:
    """
    Check if two photos are visually diverse (different enough).

    Photos are considered diverse if their similarity is below the threshold.

    Args:
        photo_a: Path to first photo
        photo_b: Path to second photo
        threshold: Similarity threshold (0.0-1.0). Default 0.85 means photos
                  with >85% similarity are considered too similar.
        method: Similarity method ("histogram")

    Returns:
        True if photos are diverse (different enough), False if too similar

    Examples:
        >>> if is_diverse(photo1, photo2, threshold=0.85):
        ...     print("Photos are different enough")
        ... else:
        ...     print("Photos are too similar (likely duplicates)")
    """
    try:
        similarity = calculate_similarity(photo_a, photo_b, method=method)
        return similarity < threshold
    except (ValueError, FileNotFoundError):
        # If comparison fails, assume they're diverse (don't reject)
        return True


def filter_similar_photos(
    photo_paths: List[Path],
    threshold: float = 0.85,
    method: str = "histogram"
) -> List[Path]:
    """
    Filter out visually similar photos from a list.

    Keeps the first photo and removes subsequent similar ones.
    Useful for removing burst-mode duplicates.

    Args:
        photo_paths: List of photo paths to filter
        threshold: Similarity threshold (0.0-1.0)
        method: Similarity method ("histogram")

    Returns:
        Filtered list with similar photos removed

    Examples:
        >>> burst_photos = [photo1, photo2, photo3, photo4]
        >>> unique_photos = filter_similar_photos(burst_photos, threshold=0.90)
        >>> print(f"Kept {len(unique_photos)}/{len(burst_photos)} photos")
    """
    if not photo_paths:
        return []

    filtered = [photo_paths[0]]  # Always keep first photo

    for candidate in photo_paths[1:]:
        # Check if candidate is diverse from all already-selected photos
        is_diverse_from_all = all(
            is_diverse(candidate, selected, threshold, method)
            for selected in filtered
        )

        if is_diverse_from_all:
            filtered.append(candidate)

    return filtered


def find_most_diverse_subset(
    photo_paths: List[Path],
    target_count: int,
    threshold: float = 0.85,
    method: str = "histogram"
) -> List[Path]:
    """
    Find the most diverse subset of photos.

    Greedily selects photos that are maximally different from each other.

    Args:
        photo_paths: List of photo paths to select from
        target_count: How many photos to select
        threshold: Similarity threshold (0.0-1.0)
        method: Similarity method ("histogram")

    Returns:
        List of diverse photos (up to target_count)

    Examples:
        >>> candidates = [photo1, photo2, photo3, photo4, photo5]
        >>> diverse = find_most_diverse_subset(candidates, target_count=3)
        >>> print(f"Selected {len(diverse)} diverse photos")
    """
    if not photo_paths or target_count <= 0:
        return []

    if len(photo_paths) <= target_count:
        return photo_paths

    # Start with first photo
    selected = [photo_paths[0]]
    remaining = list(photo_paths[1:])

    while len(selected) < target_count and remaining:
        # Find photo most different from already selected
        best_candidate = None
        best_min_similarity = float('inf')

        for candidate in remaining:
            # Calculate minimum similarity to any selected photo
            similarities = [
                calculate_similarity(candidate, selected_photo, method)
                for selected_photo in selected
            ]
            min_similarity = min(similarities) if similarities else 0.0

            # Want photo with lowest similarity to closest match
            # (i.e., most different from everything selected so far)
            if min_similarity < best_min_similarity:
                best_min_similarity = min_similarity
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break

    return selected


def _calculate_histogram_similarity(photo_a: Path, photo_b: Path) -> float:
    """
    Calculate similarity using color histogram comparison.

    Fast and effective for detecting visually similar photos.

    Args:
        photo_a: Path to first photo
        photo_b: Path to second photo

    Returns:
        Similarity score (0.0 = different, 1.0 = identical)
    """
    # Load images
    img_a = cv2.imread(str(photo_a))
    img_b = cv2.imread(str(photo_b))

    if img_a is None or img_b is None:
        raise ValueError("Failed to load one or both images")

    # Convert to HSV for better color comparison
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)

    # Calculate histograms for each channel
    hist_a = []
    hist_b = []

    # H channel: 180 bins (0-179 in OpenCV)
    # S channel: 256 bins
    # V channel: 256 bins
    channels = [(0, 180), (1, 256), (2, 256)]

    for channel_idx, bins in channels:
        h_a = cv2.calcHist([hsv_a], [channel_idx], None, [bins], [0, bins])
        h_b = cv2.calcHist([hsv_b], [channel_idx], None, [bins], [0, bins])

        # Normalize histograms
        h_a = cv2.normalize(h_a, h_a).flatten()
        h_b = cv2.normalize(h_b, h_b).flatten()

        hist_a.append(h_a)
        hist_b.append(h_b)

    # Compare histograms using correlation (range: -1 to 1, higher = more similar)
    similarities = []
    for h_a, h_b in zip(hist_a, hist_b):
        # Use correlation method (returns 1 for perfect match)
        correlation = cv2.compareHist(
            h_a.reshape(-1, 1),
            h_b.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        similarities.append(correlation)

    # Average similarity across channels
    # Convert from [-1, 1] to [0, 1] range
    avg_similarity = np.mean(similarities)
    normalized = (avg_similarity + 1.0) / 2.0

    return max(0.0, min(1.0, normalized))


def calculate_similarity_matrix(photo_paths: List[Path], method: str = "histogram") -> np.ndarray:
    """
    Calculate pairwise similarity matrix for a list of photos.

    Useful for analyzing diversity across entire set.

    Args:
        photo_paths: List of photo paths
        method: Similarity method ("histogram")

    Returns:
        NxN similarity matrix where element [i,j] is similarity between photo i and j

    Examples:
        >>> matrix = calculate_similarity_matrix(photos)
        >>> avg_similarity = np.mean(matrix)
        >>> print(f"Average similarity: {avg_similarity:.2f}")
    """
    n = len(photo_paths)
    matrix = np.zeros((n, n))

    for i in range(n):
        matrix[i, i] = 1.0  # Photo is identical to itself

        for j in range(i + 1, n):
            try:
                similarity = calculate_similarity(photo_paths[i], photo_paths[j], method)
                matrix[i, j] = similarity
                matrix[j, i] = similarity  # Symmetric
            except (ValueError, FileNotFoundError):
                # If comparison fails, assume they're different
                matrix[i, j] = 0.0
                matrix[j, i] = 0.0

    return matrix


def get_diversity_stats(photo_paths: List[Path], method: str = "histogram") -> dict:
    """
    Calculate diversity statistics for a set of photos.

    Args:
        photo_paths: List of photo paths
        method: Similarity method ("histogram")

    Returns:
        Dictionary with diversity statistics:
        {
            'avg_similarity': float,
            'min_similarity': float,
            'max_similarity': float,
            'diversity_score': float (0-1, higher = more diverse),
            'similar_pairs': int (count of pairs with >85% similarity)
        }

    Examples:
        >>> stats = get_diversity_stats(selected_photos)
        >>> print(f"Diversity score: {stats['diversity_score']:.2f}")
        >>> print(f"Similar pairs: {stats['similar_pairs']}")
    """
    if len(photo_paths) < 2:
        return {
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'diversity_score': 1.0,
            'similar_pairs': 0
        }

    matrix = calculate_similarity_matrix(photo_paths, method)

    # Get upper triangle (excluding diagonal)
    upper_triangle = matrix[np.triu_indices_from(matrix, k=1)]

    avg_sim = float(np.mean(upper_triangle))
    min_sim = float(np.min(upper_triangle))
    max_sim = float(np.max(upper_triangle))

    # Diversity score: inverse of similarity (1.0 = perfectly diverse)
    diversity = 1.0 - avg_sim

    # Count similar pairs (>85% similar)
    similar_pairs = int(np.sum(upper_triangle > 0.85))

    return {
        'avg_similarity': avg_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'diversity_score': diversity,
        'similar_pairs': similar_pairs
    }
