"""Uniform colorscheme."""


class Uniform():
    """Uniform colorscheme."""

    def __init__(self, color=(0., 0., 1.)):
        """Constructor.
        
        Args:
            color: 3-tuple of floats in [0, 1]. RGB color for all faces.
        """
        self._color = color

    def __call__(self, puzzle):
        colors = {
            face: self._color for face in puzzle.faces
        }
        return colors
