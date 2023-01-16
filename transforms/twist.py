"""Random twist transform."""

from . import base_transform
import logging
import numpy as np


class Twist(base_transform.BaseTransform):
    """Twist class.
    
    A twist takes a point and locally twists space around that point. The twist
    has an exponentially decaying basin that is draw into the rotation. It has
    the effect of spinning a fork in a bowl of honey: Points closer to the twist
    center will rotate more than points further away.
    """

    def __init__(self,
                 puzzle_bound,
                 density=0.,
                 rotation_magnitude=0.,
                 basin_tau_baseline=0.,
                 basin_tau_variance=0.):
        """Constructor.
        
        Args:
            puzzle_bound: Sclar. Maximum radius of the puzzle.
            density: Scalar. Density of twists in units of twises per square
                unit of space.
            rotation_magnitude: Scalar. Maximum magnitude of twist rotations in
                radians. Twist magnitudes are sampled uniformly from
                [-rotation_magnitude, rotation_magnitude].
            basin_tau_baseline: Minimum value of twist basin magnitudes.
            basin_tau_variance: Variance of the twist basin magnitudes.
        """
        logging.info('Constructing Twist transform')
        self._puzzle_bound = puzzle_bound
        self._density = density
        self._rotation_magnitude = rotation_magnitude
        self._basin_tau_variance = basin_tau_variance
        self._basin_tau_baseline = basin_tau_baseline

        self._num_twists = self._get_num_twists()
        logging.info(f'num_twists = {self._num_twists}')
        self._twist_centers = self._get_twist_centers()
        self._twist_angles = self._get_twist_angles()
        self._twist_basin_taus = self._get_twist_basin_taus()

    def _get_num_twists(self):
        volume = self._puzzle_bound ** 2
        num_twists = int(np.round(self._density * volume))
        return num_twists

    def _get_twist_centers(self):
        centers = np.random.uniform(
            -self._puzzle_bound, self._puzzle_bound, size=(self._num_twists, 2))
        return centers
    
    def _get_twist_angles(self):
        twist_angles = np.random.uniform(
            -self._rotation_magnitude, self._rotation_magnitude,
            size=self._num_twists)
        return twist_angles

    def _get_twist_basin_taus(self):
        twist_basin_taus = self._basin_tau_baseline + np.random.exponential(
            self._basin_tau_variance, size=self._num_twists)
        return twist_basin_taus
    
    def transform(self, points):
        perturbations = np.zeros_like(points)
        for c, a, t in zip(self._twist_centers,
                           self._twist_angles,
                           self._twist_basin_taus):
            relative_diffs = points - c[np.newaxis]
            relative_dists = np.linalg.norm(relative_diffs, axis=1)
            angles = a * np.exp(-t * relative_dists)
            perturbations += relative_dists[:, np.newaxis] * np.stack(
                [1 - np.cos(angles), np.sin(angles)], axis=1)
            
        perturbed_points = points + perturbations
        return perturbed_points
