"""
Create synthetic test fixtures for emotional significance testing.

This script creates simple test images that can be used for integration testing.
"""

import cv2
import numpy as np
from pathlib import Path


def create_synthetic_face_image(output_path: Path, num_faces: int = 1):
    """
    Create a synthetic image with simple face-like shapes.

    Args:
        output_path: Where to save the image
        num_faces: Number of faces to draw
    """
    # Create white background
    height, width = 600, 800
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw face-like ellipses
    face_width = 120
    face_height = 150

    if num_faces == 1:
        # Single centered face
        center = (width // 2, height // 2)
        cv2.ellipse(image, center, (face_width, face_height), 0, 0, 360, (200, 180, 150), -1)
        # Eyes
        cv2.circle(image, (center[0] - 30, center[1] - 20), 10, (50, 50, 50), -1)
        cv2.circle(image, (center[0] + 30, center[1] - 20), 10, (50, 50, 50), -1)
        # Smile
        cv2.ellipse(image, (center[0], center[1] + 30), (40, 20), 0, 0, 180, (50, 50, 50), 2)

    elif num_faces == 2:
        # Two faces side by side
        centers = [(width // 3, height // 2), (2 * width // 3, height // 2)]
        for center in centers:
            cv2.ellipse(image, center, (face_width, face_height), 0, 0, 360, (200, 180, 150), -1)
            cv2.circle(image, (center[0] - 30, center[1] - 20), 10, (50, 50, 50), -1)
            cv2.circle(image, (center[0] + 30, center[1] - 20), 10, (50, 50, 50), -1)
            cv2.ellipse(image, (center[0], center[1] + 30), (40, 20), 0, 0, 180, (50, 50, 50), 2)

    elif num_faces >= 3:
        # Multiple faces in a grid
        cols = min(3, num_faces)
        rows = (num_faces + cols - 1) // cols

        spacing_x = width // (cols + 1)
        spacing_y = height // (rows + 1)

        for i in range(num_faces):
            row = i // cols
            col = i % cols
            center = ((col + 1) * spacing_x, (row + 1) * spacing_y)

            cv2.ellipse(image, center, (face_width // 2, face_height // 2), 0, 0, 360, (200, 180, 150), -1)
            cv2.circle(image, (center[0] - 15, center[1] - 10), 5, (50, 50, 50), -1)
            cv2.circle(image, (center[0] + 15, center[1] - 10), 5, (50, 50, 50), -1)
            cv2.ellipse(image, (center[0], center[1] + 15), (20, 10), 0, 0, 180, (50, 50, 50), 2)

    # Save image
    cv2.imwrite(str(output_path), image)
    print(f"Created: {output_path}")


def create_landscape_image(output_path: Path):
    """Create a landscape image with no faces."""
    height, width = 600, 800

    # Blue sky gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height // 2):
        image[y, :] = [200 - y // 3, 220 - y // 3, 255]

    # Green grass
    for y in range(height // 2, height):
        image[y, :] = [100, 180, 50]

    # Simple mountain
    pts = np.array([[width // 4, height // 2],
                    [width // 2, height // 4],
                    [3 * width // 4, height // 2]], np.int32)
    cv2.fillPoly(image, [pts], (120, 120, 120))

    cv2.imwrite(str(output_path), image)
    print(f"Created: {output_path}")


if __name__ == '__main__':
    fixtures_dir = Path(__file__).parent / 'fixtures'
    fixtures_dir.mkdir(exist_ok=True)

    # Create test images
    create_synthetic_face_image(fixtures_dir / 'single_face.jpg', num_faces=1)
    create_synthetic_face_image(fixtures_dir / 'couple.jpg', num_faces=2)
    create_synthetic_face_image(fixtures_dir / 'group.jpg', num_faces=5)
    create_landscape_image(fixtures_dir / 'landscape.jpg')

    print("\nTest fixtures created successfully!")
