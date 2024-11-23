"""Randomly dilate and/or contract holes."""

from . import base_transform
import logging
import numpy as np


def exponential_dilation_tau_fn(baseline=0., eccentricity_tau=0.):
    """Dilation magnitude sampled from an exponential distribution of radius.

    Args:
        baseline: Scalar. Value of dilation magnitude at radius 0.
        eccentricity_tau: Scalar. Scale of the exponential function of radius
            that the dilation magnitude is sampled from.

    Returns:
        _get_dilation_tau: Function taking vector of hole eccentricities and
            returning hole magnitudes.
    """
    def _get_dilation_tau(eccentricities):
        tau = baseline + np.random.exponential(
            eccentricity_tau * eccentricities)
        return tau
    return _get_dilation_tau


def normal_dilation_tau_fn(loc_intercept=0.,
                           scale_intercept=0.,
                           loc_slope=0.,
                           scale_slope=0.,
                           min_dilation_tau=0.):
    """Dilation magnitude sampled from a radially-modulated normal distribution.

    The mean and standard deviation of the normal distribution are linear
    functions of radius. Depending on how you choose these linear functions,
    this can give the effect of larger and more variable hole dilations as you
    move away from the puzzle center.

    Args:
        loc_intercept: Scalar. Value of the mean of the normal at radius 0.
        scale_intercept: Scalar. Value of the scale of the normal at radius 0.
        loc_slope: Scalar. Scaling of the mean of the normal as a function of
            radius.
        scale_slope: Scalar. Scaling of the scale of the normal as a function of
            radius.
        min_dilation_tau: Scalar. Minimum value of the dilation magnitude.

    Returns:
        _get_dilation_tau: Function taking vector of hole eccentricities and
            returning hole magnitudes.
    """
    def _get_dilation_tau(eccentricities):
        loc = loc_intercept + loc_slope * eccentricities
        scale = scale_intercept + scale_slope * eccentricities
        tau = np.random.normal(loc=loc, scale=scale)
        tau[tau < min_dilation_tau] = min_dilation_tau
        return tau
    return _get_dilation_tau


class DilateHoles(base_transform.BaseTransform):
    """DilateHoles class."""

    def __init__(self, dilation_tau_fn, puzzle):
        """Constructor.
        
        Args:
            dilation_tau_fn: Function that takes a vector of hole eccentricities
                and returns a hole dilation magnitude.
            puzzle: Puzzle object. Must be an instance of
                ../base_puzzle.BasePuzzle.
        """
        logging.info('Constructing DilateHoles transform')

        self._dilation_tau_fn = dilation_tau_fn
        self._puzzle = puzzle

        self._dilation_centers = np.array([h['center'] for h in puzzle.holes])
        hole_eccentricities = np.linalg.norm(self._dilation_centers, axis=1)
        self._dilation_taus = self._dilation_tau_fn(hole_eccentricities)
    
    def transform(self, points):
        """Apply transform to points.
        
        Args:
            points: Size [_, 2] float array of coordinates of points to
                transform.
        
        Returns:
            perturbed_points: Size [_, 2] float array of coordinates of points
                after applying this transform that dilates holes in the puzzle.
        """
        perturbations = np.zeros_like(points)
        for c, t in zip(self._dilation_centers, self._dilation_taus):
            relative_diffs = points - c[np.newaxis]
            relative_dists = np.linalg.norm(
                relative_diffs, axis=1, keepdims=True)
            perturbation_magnitudes = t * np.exp(-1 * relative_dists)
            perturbations += perturbation_magnitudes * relative_diffs
            
        perturbed_points = points + perturbations
        return perturbed_points
