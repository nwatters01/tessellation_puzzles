"""Bubbles colorscheme.

Note: This colorscheme assumes a continuous color palette.

This assigns colors based on distance to bubble centers, where bubble centers
are random points in the puzzle.
"""

import colorsys
import numpy as np


def _get_face_centroid(face, puzzle):
    vertices = [puzzle.vertices[v] for v in face]
    centroid = np.mean(vertices, axis=0)
    return centroid


def _color_value_to_rgb(color_value):
    hue = 0.7 * (1. - color_value)
    value = 1. - 0.7 * (1. - color_value)
    rgb = colorsys.hsv_to_rgb(hue, 1., value)
    return rgb


class Bubbles():
    """Bubbles colorscheme."""

    def __init__(self, bubble_density):
        """Constructor.
        
        Args:
            bubble_density: Density of bubbles.
        """
        self._bubble_density = bubble_density

    def _get_color_value(self, face, puzzle, bubble_centers):
        centroid = _get_face_centroid(face, puzzle)
        dists_to_centers = np.linalg.norm(
            bubble_centers - centroid[np.newaxis], axis=1)
        min_dist_to_center = np.min(dists_to_centers)
        color_value = -1. * min_dist_to_center
        return color_value

    def __call__(self, puzzle):
        radial_bound = puzzle.get_bound()
        num_bubbles = int(np.round(
            self._bubble_density * 2 * radial_bound * radial_bound))
        bubble_centers = np.random.uniform(
            -radial_bound, radial_bound, size=(num_bubbles, 2))

        # Get color values for each face
        color_values = np.array([
            self._get_color_value(face, puzzle, bubble_centers)
            for face in puzzle.faces
        ])

        # Normalize color values
        min_color_value = np.min(color_values)
        max_color_value = np.max(color_values)
        color_values = (color_values - min_color_value) / (
            max_color_value - min_color_value)

        # Convert color values ro RGB
        colors = {
            face: _color_value_to_rgb(color_value)
            for face, color_value in zip(puzzle.faces, color_values)
        }
        return colors
