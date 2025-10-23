"""
The Twelve Curation Engine - Core curator implementation.

Automatically selects the best 12 photos from a year's library using:
- Combined quality + emotional scoring
- Temporal diversity (spread across months)
- Visual diversity (avoid near-duplicates)
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .data_classes import PhotoCandidate, TwelveSelection, CurationConfig
from .exif_utils import extract_datetime, get_month
from .diversity import is_diverse, get_diversity_stats

# Import analyzers from sibling modules
import sys
from pathlib import Path as PathLib
src_path = PathLib(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from photo_quality_analyzer import PhotoQualityAnalyzer
from emotional_significance import EmotionalAnalyzer


class TwelveCurator:
    """
    The Twelve Curation Engine.

    Selects exactly 12 photos from a year's library based on quality,
    emotional significance, temporal diversity, and visual variety.

    Examples:
        >>> curator = TwelveCurator()
        >>> selection = curator.curate_year(photo_paths, year=2024)
        >>> print(selection.summary())
        >>>
        >>> # With custom config
        >>> config = CurationConfig.people_first()
        >>> curator = TwelveCurator(config)
        >>> selection = curator.curate_year(photo_paths, year=2024)
    """

    def __init__(self, config: Optional[CurationConfig] = None):
        """
        Initialize the curator.

        Args:
            config: Curation configuration. Defaults to balanced strategy.
        """
        self.config = config or CurationConfig.balanced()
        self.quality_analyzer = PhotoQualityAnalyzer()
        self.emotional_analyzer = EmotionalAnalyzer()

    def curate_year(
        self,
        photo_paths: List[Path],
        year: int,
        strategy: Optional[str] = None
    ) -> TwelveSelection:
        """
        Curate 12 photos from a year's library.

        Main curation algorithm:
        1. Score all photos (quality + emotional)
        2. Group by month
        3. Select best from each month
        4. Apply visual diversity filter
        5. Return exactly 12 (or fewer if <12 available)

        Args:
            photo_paths: List of photo paths to curate from
            year: Year being curated
            strategy: Optional strategy override (balanced, aesthetic_first, people_first, top_heavy)

        Returns:
            TwelveSelection with exactly 12 photos (or fewer if <12 available)

        Examples:
            >>> selection = curator.curate_year(photos, year=2024)
            >>> print(f"Selected {len(selection.photos)} photos")
            >>> print(f"Avg score: {selection.stats['avg_combined']:.1f}")
        """
        # Apply strategy override if provided
        if strategy:
            self.config.strategy = strategy

        # Phase 1: Score all photos
        candidates = self._score_all_photos(photo_paths, year)

        if not candidates:
            # No photos found - return empty selection
            return self._create_empty_selection(year, 0)

        if len(candidates) <= 12:
            # Fewer than 12 photos - return all
            return self._create_selection(year, candidates, candidates)

        # Phase 2: Temporal distribution
        temporal_selections = self._apply_temporal_distribution(candidates)

        # Phase 3: Visual diversity
        if self.config.enable_diversity_filter:
            final_selections = self._apply_visual_diversity(temporal_selections)
        else:
            final_selections = temporal_selections[:12]

        # Phase 4: Ensure exactly 12 (fill if needed)
        if len(final_selections) < 12:
            final_selections = self._fill_to_twelve(final_selections, candidates)

        # Phase 5: Create final selection
        return self._create_selection(year, final_selections[:12], candidates)

    def curate_from_source(
        self,
        photo_source,  # PhotoSource interface
        year: int,
        strategy: Optional[str] = None,
        progress_callback = None
    ) -> TwelveSelection:
        """
        Curate 12 photos from a PhotoSource (local or cloud).

        This is the new unified method that works with any PhotoSource
        implementation (LocalPhotoSource, GooglePhotosSource, etc.).

        Main curation algorithm:
        1. Scan photos from source
        2. Score all photos (quality + emotional)
        3. Group by month
        4. Select best from each month
        5. Apply visual diversity filter
        6. Return exactly 12 (or fewer if <12 available)

        Args:
            photo_source: PhotoSource instance (local or cloud)
            year: Year being curated
            strategy: Optional strategy override (balanced, aesthetic_first, people_first, top_heavy)
            progress_callback: Optional callback(current, total, status_msg)

        Returns:
            TwelveSelection with exactly 12 photos (or fewer if <12 available)

        Examples:
            >>> from photo_sources import PhotoSourceFactory
            >>> source = PhotoSourceFactory.create_google_photos('creds.json')
            >>> source.authenticate()
            >>> selection = curator.curate_from_source(source, year=2024)
            >>> print(f"Selected {len(selection.photos)} photos")

        Note: Automatically calls source.cleanup() after curation
        """
        # Apply strategy override if provided
        if strategy:
            self.config.strategy = strategy

        try:
            # Phase 1: Scan and score all photos from source
            candidates = self._score_photos_from_source(
                photo_source,
                year,
                progress_callback
            )

            if not candidates:
                # No photos found - return empty selection
                return self._create_empty_selection(year, 0)

            if len(candidates) <= 12:
                # Fewer than 12 photos - return all
                return self._create_selection(year, candidates, candidates)

            # Phase 2: Temporal distribution
            temporal_selections = self._apply_temporal_distribution(candidates)

            # Phase 3: Visual diversity
            if self.config.enable_diversity_filter:
                final_selections = self._apply_visual_diversity(temporal_selections)
            else:
                final_selections = temporal_selections[:12]

            # Phase 4: Ensure exactly 12 (fill if needed)
            if len(final_selections) < 12:
                final_selections = self._fill_to_twelve(final_selections, candidates)

            # Phase 5: Create final selection
            return self._create_selection(year, final_selections[:12], candidates)

        finally:
            # Always cleanup source (temp files, etc.)
            photo_source.cleanup()

    def preview_candidates(
        self,
        photo_paths: List[Path],
        year: int,
        top_n: int = 50
    ) -> List[PhotoCandidate]:
        """
        Get top N candidates before final selection.

        Useful for previewing what photos are being considered.

        Args:
            photo_paths: List of photo paths
            year: Year to filter by
            top_n: How many top candidates to return

        Returns:
            List of top candidates sorted by combined score

        Examples:
            >>> candidates = curator.preview_candidates(photos, 2024, top_n=20)
            >>> for c in candidates[:5]:
            ...     print(f"{c.photo_path.name}: {c.combined_score:.1f}")
        """
        candidates = self._score_all_photos(photo_paths, year)
        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates[:top_n]

    def curate_with_fallback(
        self,
        photo_paths: List[Path],
        year: int
    ) -> TwelveSelection:
        """
        Curate with automatic fallback if constraints can't be met.

        Tries balanced strategy first, falls back to top_heavy if needed.

        Args:
            photo_paths: List of photo paths
            year: Year being curated

        Returns:
            TwelveSelection (may relax constraints to find 12 photos)
        """
        # Try normal curation first
        try:
            selection = self.curate_year(photo_paths, year)
            if len(selection.photos) >= 12:
                return selection
        except Exception:
            pass

        # Fall back to top_heavy strategy (less strict temporal requirements)
        original_strategy = self.config.strategy
        self.config.strategy = "top_heavy"
        self.config.enforce_monthly_distribution = False

        selection = self.curate_year(photo_paths, year)

        # Restore original strategy
        self.config.strategy = original_strategy

        return selection

    # ========================================================================
    # Phase 1: Scoring
    # ========================================================================

    def _score_photos_from_source(
        self,
        photo_source,  # PhotoSource interface
        year: int,
        progress_callback = None
    ) -> List[PhotoCandidate]:
        """
        Score all photos from a PhotoSource.

        Args:
            photo_source: PhotoSource instance
            year: Year to filter by
            progress_callback: Optional callback(current, total, status_msg)

        Returns:
            List of PhotoCandidate objects
        """
        candidates = []
        photo_count = 0

        # Scan photos from source (downloads if cloud source)
        for photo_path_str in photo_source.scan(year=year):
            photo_count += 1

            # Progress callback
            if progress_callback:
                progress_callback(
                    photo_count,
                    None,  # Total unknown during streaming
                    f"Analyzing photo {photo_count}..."
                )

            try:
                # Convert to Path
                photo_path = Path(photo_path_str)

                # Get metadata from source
                source_metadata = photo_source.get_metadata(photo_path_str)
                timestamp = source_metadata.get('timestamp')
                month = source_metadata.get('month')

                # Skip if wrong year (source filter may not be perfect)
                if timestamp and timestamp.year != year:
                    continue

                # Analyze quality
                quality_result = self.quality_analyzer.analyze_photo(str(photo_path))
                quality_score = quality_result.composite

                # Analyze emotional significance
                emotional_result = self.emotional_analyzer.analyze_photo(str(photo_path))
                emotional_score = emotional_result.composite

                # Calculate combined score
                combined_score = (
                    quality_score * self.config.quality_weight +
                    emotional_score * self.config.emotional_weight
                )

                # Filter by minimum score
                if combined_score < self.config.min_combined_score:
                    continue

                # Create metadata
                metadata = {
                    'quality_tier': quality_result.tier,
                    'emotional_tier': emotional_result.tier,
                    'has_faces': emotional_result.face_count > 0,
                    'face_count': emotional_result.face_count,
                    'has_positive_emotion': emotional_result.has_positive_emotion,
                    'original_url': photo_source.get_original_url(photo_path_str),  # NEW: Google Photos URL
                    'source_metadata': source_metadata
                }

                candidate = PhotoCandidate(
                    photo_path=photo_path,
                    timestamp=timestamp,
                    month=month,
                    quality_score=quality_score,
                    emotional_score=emotional_score,
                    combined_score=combined_score,
                    metadata=metadata
                )
                candidates.append(candidate)

            except Exception as e:
                # Skip photos that fail to analyze
                print(f"Warning: Failed to analyze {photo_path_str}: {e}")
                continue

        return candidates

    def _score_all_photos(self, photo_paths: List[Path], year: int) -> List[PhotoCandidate]:
        """
        Score all photos and create candidates.

        Args:
            photo_paths: List of photo paths
            year: Year to filter by

        Returns:
            List of PhotoCandidate objects
        """
        candidates = []

        for photo_path in photo_paths:
            try:
                candidate = self._score_single_photo(photo_path, year)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                # Skip photos that fail to analyze
                continue

        return candidates

    def _score_single_photo(self, photo_path: Path, year: int) -> Optional[PhotoCandidate]:
        """
        Score a single photo and create a candidate.

        Args:
            photo_path: Path to photo
            year: Expected year

        Returns:
            PhotoCandidate or None if photo doesn't meet criteria
        """
        # Extract timestamp
        timestamp = extract_datetime(photo_path)
        month = timestamp.month if timestamp else None

        # Filter by year if timestamp available
        if timestamp and timestamp.year != year:
            return None

        # Analyze quality
        quality_result = self.quality_analyzer.analyze_photo(str(photo_path))
        quality_score = quality_result.composite

        # Analyze emotional significance
        emotional_result = self.emotional_analyzer.analyze_photo(str(photo_path))
        emotional_score = emotional_result.composite

        # Calculate combined score
        combined_score = (
            quality_score * self.config.quality_weight +
            emotional_score * self.config.emotional_weight
        )

        # Filter by minimum score
        if combined_score < self.config.min_combined_score:
            return None

        # Create metadata
        metadata = {
            'quality_tier': quality_result.tier,
            'emotional_tier': emotional_result.tier,
            'has_faces': emotional_result.face_count > 0,
            'face_count': emotional_result.face_count,
            'has_positive_emotion': emotional_result.has_positive_emotion
        }

        return PhotoCandidate(
            photo_path=photo_path,
            timestamp=timestamp,
            month=month,
            quality_score=quality_score,
            emotional_score=emotional_score,
            combined_score=combined_score,
            metadata=metadata
        )

    # ========================================================================
    # Phase 2: Temporal Distribution
    # ========================================================================

    def _apply_temporal_distribution(self, candidates: List[PhotoCandidate]) -> List[PhotoCandidate]:
        """
        Apply temporal distribution strategy.

        Args:
            candidates: List of all candidates

        Returns:
            List of candidates selected for temporal diversity
        """
        if self.config.strategy == "top_heavy":
            # Top-heavy: Just take best scores
            sorted_candidates = sorted(candidates, key=lambda c: c.combined_score, reverse=True)
            return sorted_candidates[:12]

        # Balanced, people_first, aesthetic_first: Distribute across months
        return self._select_by_month(candidates)

    def _select_by_month(self, candidates: List[PhotoCandidate]) -> List[PhotoCandidate]:
        """
        Select photos distributed across months.

        Args:
            candidates: List of all candidates

        Returns:
            List of candidates with temporal diversity
        """
        # Group by month
        by_month = defaultdict(list)
        no_month = []

        for candidate in candidates:
            if candidate.month:
                by_month[candidate.month].append(candidate)
            else:
                no_month.append(candidate)

        # Sort photos within each month by score
        for month_photos in by_month.values():
            month_photos.sort(key=lambda c: c.combined_score, reverse=True)

        # Select best from each month
        selections = []
        for month in range(1, 13):
            if month in by_month and by_month[month]:
                selections.append(by_month[month][0])

        # If we have 12, we're done
        if len(selections) >= 12:
            # Sort by score and take top 12
            selections.sort(key=lambda c: c.combined_score, reverse=True)
            return selections[:12]

        # Need more photos - add next best from each month
        while len(selections) < 12:
            added = False

            # Try to add second-best from each month
            for month in range(1, 13):
                if len(selections) >= 12:
                    break

                month_photos = by_month.get(month, [])
                # Find photos from this month not yet selected
                available = [p for p in month_photos if p not in selections]

                if available:
                    selections.append(available[0])
                    added = True

            if not added:
                # No more photos with months - add from no_month or break
                if no_month:
                    no_month.sort(key=lambda c: c.combined_score, reverse=True)
                    selections.extend(no_month[:12 - len(selections)])
                break

        return selections[:12]

    # ========================================================================
    # Phase 3: Visual Diversity
    # ========================================================================

    def _apply_visual_diversity(self, candidates: List[PhotoCandidate]) -> List[PhotoCandidate]:
        """
        Apply visual diversity filter to remove near-duplicates.

        Args:
            candidates: List of candidates to filter

        Returns:
            List of diverse candidates
        """
        if len(candidates) <= 12:
            return candidates

        # Sort by combined score (highest first)
        candidates_sorted = sorted(candidates, key=lambda c: c.combined_score, reverse=True)

        # Greedily select diverse photos
        diverse_selections = []

        for candidate in candidates_sorted:
            # Check if diverse from all already selected
            is_diverse_from_all = all(
                is_diverse(
                    candidate.photo_path,
                    selected.photo_path,
                    threshold=self.config.visual_diversity_threshold
                )
                for selected in diverse_selections
            )

            if is_diverse_from_all:
                diverse_selections.append(candidate)

            # Stop when we have 12
            if len(diverse_selections) >= 12:
                break

        return diverse_selections

    # ========================================================================
    # Phase 4: Fill to Twelve
    # ========================================================================

    def _fill_to_twelve(
        self,
        selections: List[PhotoCandidate],
        all_candidates: List[PhotoCandidate]
    ) -> List[PhotoCandidate]:
        """
        Fill selections to reach 12 photos.

        If diversity filtering removed too many, add back top scorers.

        Args:
            selections: Current selections (< 12)
            all_candidates: All available candidates

        Returns:
            List with up to 12 photos
        """
        if len(selections) >= 12:
            return selections

        # Get candidates not yet selected
        remaining = [c for c in all_candidates if c not in selections]

        # Sort by score
        remaining.sort(key=lambda c: c.combined_score, reverse=True)

        # Add top scorers until we have 12
        needed = 12 - len(selections)
        selections.extend(remaining[:needed])

        return selections

    # ========================================================================
    # Result Creation
    # ========================================================================

    def _create_selection(
        self,
        year: int,
        selected: List[PhotoCandidate],
        all_candidates: List[PhotoCandidate]
    ) -> TwelveSelection:
        """
        Create final TwelveSelection result.

        Args:
            year: Year being curated
            selected: Selected photos
            all_candidates: All candidates considered

        Returns:
            TwelveSelection object
        """
        # Sort by month, then by score
        selected_sorted = sorted(
            selected,
            key=lambda c: (c.month if c.month else 999, -c.combined_score)
        )

        # Calculate statistics
        stats = self._calculate_stats(selected_sorted, all_candidates)

        return TwelveSelection(
            year=year,
            photos=selected_sorted,
            strategy=self.config.strategy,
            stats=stats,
            created_at=datetime.now()
        )

    def _create_empty_selection(self, year: int, total_candidates: int) -> TwelveSelection:
        """Create an empty selection when no photos found."""
        return TwelveSelection(
            year=year,
            photos=[],
            strategy=self.config.strategy,
            stats={
                'total_candidates': total_candidates,
                'avg_quality': 0.0,
                'avg_emotional': 0.0,
                'avg_combined': 0.0,
                'months_represented': 0,
                'photos_with_faces': 0,
                'photos_with_smiles': 0,
                'diversity_score': 0.0
            },
            created_at=datetime.now()
        )

    def _calculate_stats(
        self,
        selected: List[PhotoCandidate],
        all_candidates: List[PhotoCandidate]
    ) -> dict:
        """
        Calculate statistics for the selection.

        Args:
            selected: Selected photos
            all_candidates: All candidates

        Returns:
            Dictionary of statistics
        """
        if not selected:
            return {
                'total_candidates': len(all_candidates),
                'avg_quality': 0.0,
                'avg_emotional': 0.0,
                'avg_combined': 0.0,
                'months_represented': 0,
                'photos_with_faces': 0,
                'photos_with_smiles': 0,
                'diversity_score': 0.0
            }

        # Basic score statistics
        quality_scores = [c.quality_score for c in selected]
        emotional_scores = [c.emotional_score for c in selected]
        combined_scores = [c.combined_score for c in selected]

        # Temporal statistics
        months = set(c.month for c in selected if c.month)

        # Face statistics
        photos_with_faces = sum(1 for c in selected if c.metadata.get('has_faces', False))
        photos_with_smiles = sum(1 for c in selected if c.metadata.get('has_smiles', False))

        # Diversity statistics
        diversity_score = 0.0
        if len(selected) >= 2:
            try:
                diversity_stats = get_diversity_stats([c.photo_path for c in selected])
                diversity_score = diversity_stats['diversity_score']
            except Exception:
                pass

        return {
            'total_candidates': len(all_candidates),
            'avg_quality': float(np.mean(quality_scores)),
            'avg_emotional': float(np.mean(emotional_scores)),
            'avg_combined': float(np.mean(combined_scores)),
            'months_represented': len(months),
            'photos_with_faces': photos_with_faces,
            'photos_with_smiles': photos_with_smiles,
            'diversity_score': diversity_score,
            'min_quality': float(np.min(quality_scores)),
            'max_quality': float(np.max(quality_scores)),
            'min_emotional': float(np.min(emotional_scores)),
            'max_emotional': float(np.max(emotional_scores)),
            'min_combined': float(np.min(combined_scores)),
            'max_combined': float(np.max(combined_scores))
        }
