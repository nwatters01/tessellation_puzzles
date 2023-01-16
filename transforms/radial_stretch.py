"""Radially stretch space."""

from . import base_transform
import logging
import numpy as np


class RadialStretch(base_transform.BaseTransform):
    """RadialStretch class."""

    def __init__(self, tau=0.):
        """Constructor.
        
        Args:
            tau: Exponential scale of the stretch as a function of radius.
                Larger values mean space is scaled up faster as you move away
                from the puzzle center.
        """
        logging.info('Constructing RadialStretch transform')
        self._tau = tau

    def transform(self, points):
        radii = np.linalg.norm(points, axis=1)
        scalings = 1 + np.exp(self._tau * radii)
        scaled_points = scalings[:, np.newaxis] * points
        return scaled_points
