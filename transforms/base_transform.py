"""Base transform class.

A transform is a callable object that perturbs points in space. Transforms are
used for adjust the geometry of a puzzle, such as by dilating holes, warping
space, etc.
"""

import abc
import logging
import numpy as np


class BaseTransform(abc.ABC):
    """Base transform class."""

    @abc.abstractmethod
    def transform(self, points):
        """Transform function perturbing points in space.
        
        Args:
            points: Size [_, 2] float array containing coordinates of points in
                space to transform. Typically these are edgepoints of vertices
                of the puzzle.

        Returns:
            Size [_, 2] float array containing the coordinates of the input
                points after applying the transform function.
        """
        raise NotImplementedError

    def __call__(self, puzzle):
        """Apply transform to a puzzle."""
        logging.info('    Applying transform')

        # Apply transform to vertices
        puzzle.vertices = self.transform(puzzle.vertices)
        
        # Apply transform to holes
        holes = puzzle.holes
        for h in holes:
            h['center'] = self.transform(np.array([h['center']]))[0]

        # Apply transform to mesh
        mesh = puzzle.mesh
        for k, v in mesh.items():
            mesh[k] = self.transform(v)
