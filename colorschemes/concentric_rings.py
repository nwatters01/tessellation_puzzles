"""Concentric rings colorscheme."""

import numpy as np


def _get_radius(face, puzzle):
    vertices = [puzzle.vertices[v] for v in face]
    centroid = np.mean(vertices, axis=0)
    radius = np.linalg.norm(centroid)
    return radius
    

class ConcentricRings():
    """Concentric rings colorscheme."""

    def __init__(self, color_per_ring, ring_thickness):
        """Constructor.
        
        Args:
            color_per_ring: List of colors. Each color is a 3-tuple of floats
                in [0, 1], an RGB color.
            ring_thickness: Scalar. Thickness of each ring.
        """
        self._color_per_ring = color_per_ring
        self._ring_thickness = ring_thickness
        
        self._num_colors = len(color_per_ring)

    def _get_color(self, face, puzzle):
        radius = _get_radius(face, puzzle)
        ring_index = int(np.round(radius / self._ring_thickness))
        color_index = ring_index % self._num_colors
        color = self._color_per_ring[color_index]
        return color

    def __call__(self, puzzle):
        colors = {
            face: self._get_color(face, puzzle) for face in puzzle.faces
        }
        return colors
