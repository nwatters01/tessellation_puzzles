"""Warp edges.

This file contains code for warping edges of the puzzle. Warping edges
introduces folds and ox-bows that make the pieces fit together in a more
satisfying way.

Each edge of the puzzle is warped independently, subject only to rejection if
one edge intersects another after warping. The warping of an edge is done with
the following steps:
    * Sample a sequence of curvatures. These curvatures are in units of radians
        per vertex, and represent the angular change (i.e. curvature) of a path.
        This curvature sampling is done by alternating positive and negative
        curves, interleaved with brief straight segments, plus some noise.
    * Construct a path from the curvature vector by integrating a vector
        traveling with that momentary curvature through space. This gives a
        nice curvy path.
    * Flatten the path, particularly near the end-points, by taking the weighted
        average between the curvy path and a straight line from start point to
        end point. The weighting is high near the ends of the paths. The
        resulting path is roughly straight near the ends, and more curvy in the
        middle. We need this to reduce the likelihood of adjacent edges
        intersecting each other near corners.
    * Scale and orient the path to have the desired start and end points of the
        edge.
"""

import logging
import numpy as np
import time

# Maximum number of attempts for rejection sampling of non-intersecting edges,
# to avoid infinite looping.
_MAX_TRIES = int(1e3)


def get_sample_curvature_magnitude(min_curvature=9. * np.pi,
                                   max_curvature=13. * np.pi):
    """Get a sampler function that returns a curvature magnitude.
    
    Args:
        min_curvature: Float. Minimum curvature, in radians per unit space.
        max_curvature: Float. Maximum curvature, in radians per unit space.

    Returns:
        Callable returning a float curvature in radians per unit space.
    """
    return lambda: np.random.uniform(min_curvature, max_curvature)


def get_sample_curve_length(min_integral=1.15 * np.pi,
                            max_integral=1.3 * np.pi):
    """Get a sampler function that returns a length of a curve.

    The length of a curve is the number of vertices in the curve, and is sampled
    to have a resulting total curve angle (integral of curvature) within a
    desired range.
    
    Args:
        min_integral: Float. Minimum total curve angle, in radians.
        max_integral: Float. Maximum total curve angle, in radians.

    Returns:
        _sample_curve_length: Function taking in curvature magnitude and
            returning a float length of the curve.
    """
    def _sample_curve_length(curvature_magnitude):
        min_length = min_integral / curvature_magnitude
        max_length = max_integral / curvature_magnitude
        return np.random.uniform(min_length, max_length)
    return _sample_curve_length


def get_sample_straight_length(min_length=0.02, max_length=0.04):
    """Get a sampler function that returns a length for a straight segment.
    
    Args:
        min_length: Minimum length, in units of space.
        max_length: Maximum length, in units of space.

    Returns:
        Callable returning a float length of a straight segment.
    """
    return lambda: np.random.uniform(min_length, max_length)


def get_sample_noise(kernel_half_width=20, noise_amplitude=30.):
    """Get a sampler function that returns noise to add to a curvature vector.

    The noise is a smoothed and centered Brownian motion vector.

    Args:
        kernel_half_width: Half-width of a tooth function smoothing kernel.
        noise_amplitude: Integral of the smoothing kernel.

    Returns:
        _sample_noise: Function taking in length of the curvature vector and
            returning a vector of noise to add to it.
    """
    kernel = np.concatenate([
        np.arange(0., kernel_half_width),
        np.arange(kernel_half_width, 0., -1),
    ], axis=0)
    kernel /= np.sum(kernel)
    kernel *= noise_amplitude
    def _sample_noise(length):
        rands = np.random.normal(
            loc=0., scale=1., size=length + 2 * kernel_half_width - 1)
        noise_vec = np.convolve(rands, kernel, mode='valid')
        noise_vec -= np.mean(noise_vec)
        return noise_vec

    return _sample_noise


def get_flattening_fn(tau=10., flattening_multiplier=0.9):
    """Get flattening function.
    
    Flattening function is a vector of weights for a weighted average between a
    curve and a straight line from start point to end point. Flattening function
    should be zero near the start and end, so the flattened curve is linear near
    the start and end. Flattening function may be higher in the middle. Here we
    use a symmetrized truncated Gaussian PDF.

    Args:
        tau: Float. Slope of the flattening function near the ends. Higher tau
            means the flattening function drops off more sharply near the ends.
        flattening_multiplier: Float no greater than 1. Overall multiplier
            applied to the flattening function. Smaller means the flattened
            curve will be overall flatter.
    """
    def _flattening_fn(n):
        linspace = np.linspace(0., 1., n)
        endpoint_dist = np.minimum(linspace, 1. - linspace)
        output = 1. - np.exp(-1. * tau * np.square(endpoint_dist))
        output *= flattening_multiplier
        return output
    return _flattening_fn


class Curvature():
    """Curvature class.
    
    This class is used to generate curvature vectors, that can then be
    integrated into curved paths for edges.

    A curvature vector is an sequence of right turns, straight segments, left
    turns, and straight segments. This creates a wave/hairpin pattern, depending
    on the curvatures and lengths of the segments. The right and left turns have
    (approximately) constant curvature and are sampled from a distribution
    specified by sample_curvature_magnitude and sample_curve_length. The
    straight segments have (approximately) zero cuvature and are sampled from a
    distrbution of lengths sample_straight_length.
    
    There is noise added to add variation to the curvatures. The noise is
    sampled independently per-segment.
    """

    def __init__(self,
                 sample_curvature_magnitude,
                 sample_curve_length,
                 sample_straight_length,
                 sample_noise,
                 resolution=500):
        """Constructor.
        
        Args:
            sample_curvature_magnitude: Callable returning float curvature
                magnitude. See get_sample_curvature_magnitude() above.
            sample_curve_length: Function taking in curvature magnitude and
                returning curve length. See get_sample_curve_length() above.
            sample_straight_length: Callable returning straight segment length.
                See get_sample_straight_length() above.
            sample_noise: Function taking in segment length and returning noise
                to add to the segment. See get_sample_noise() above.
            resolution: Int. Number of vertices in a path.
        """
        self._sample_straight_length = sample_straight_length
        self._sample_curve_length = sample_curve_length
        self._sample_curvature_magnitude = sample_curvature_magnitude
        self._sample_noise = sample_noise
        self._resolution = resolution

        # Estimate path length envelope by sampling a bunch of curve lengths and
        # straight lengths. Path length envelope is the length of a path to
        # generate before trimming it down to resolution. We generate a longer
        # path first to avoid biased sampling of which phase of the curve we
        # start with. In other words, so that we sometimes start by curving to
        # the right, sometimes by curving to the left, and sometimes by going
        # straight.
        max_curve_length = resolution * np.max([
            sample_curve_length(sample_curvature_magnitude())
            for _ in range(100)
        ])
        max_straight_length = resolution * np.max(
            [sample_straight_length() for _ in range(100)])
        buffer_length = int(2 * (max_curve_length + max_straight_length))
        self._path_length_envelope = resolution + buffer_length

    def _sample_curve(self):
        """Sample curvatures for a curved segment."""
        curvature_magnitude = self._sample_curvature_magnitude()
        curve_length = self._sample_curve_length(curvature_magnitude)
        curve_num_points = int(np.round(self._resolution * curve_length))
        curve = curvature_magnitude * np.ones(curve_num_points)
        curve += self._sample_noise(curve_num_points)
        return curve

    def _sample_straight(self):
        """Sample curvatures for a straight segment."""
        straight_length = self._sample_straight_length()
        straight_num_points = int(np.round(self._resolution * straight_length))
        straight = np.zeros(straight_num_points)
        straight += self._sample_noise(straight_num_points)
        return straight
    
    def __call__(self):
        """Sample a curvature vector of length self._resolution.
        
        Returns:
            curvature: Float array of curvatures of size [self._resolution].
        """
        phase = 0
        total_length = 0
        segments = []
        while total_length < self._path_length_envelope:
            if phase == 0:
                curve = self._sample_curve()
                segments.append(curve)
                total_length += len(curve)
            elif phase == 2:
                curve = -1. * self._sample_curve()
                segments.append(curve)
                total_length += len(curve)
            elif phase in [1, 3]:
                straight = self._sample_straight()
                segments.append(straight)
                total_length += len(straight)
            else:
                raise ValueError(f'Invalid phase {phase}')
            phase =  (phase + 1) % 4

        # Sub-sample a random interval
        curvature = np.concatenate(segments)
        start_ind = np.random.randint(
            self._path_length_envelope - self._resolution)
        curvature = curvature[start_ind: start_ind + self._resolution]
        
        return curvature

    @property
    def resolution(self):
        return self._resolution


class WarpedPath():
    """WarpedPath class.
    
    This class generated a warped path when called. Generating a warped path
    takes these steps:
        * Sample a curvature vector.
        * Integrate the curvature vector along a path to generate a path with
            that curvature profile.
        * Flatten the path, particularly near its endpoints. This helps avoid
            intersections between nearby paths in the puzzle.
        * Reject and resample if the path intersects itself.
    """

    def __init__(self,
                 curvature_object,
                 flattening_fn,
                 path_squishing_factor=0.75,
                 self_intersecting_thresh=0.05,
                 self_intersecting_stride=2):
        """Constructor.
        
        Args:
            curvature_object: Instance of Curvature() class.
            flattening_fn: Flattening function that takes in the path length and
                returns a vector of weights for a weighted average between the
                path and a straight line from start point to end point. See
                get_flattening_fn() above.
            path_squishing_factor: Float no greater than 1. Multiplier to
                linearly contract the path perpendicular to the start-end axis.
                This is a different from the flattening function, because it
                squishes the path instead of unraveling it. This is useful also
                for avoiding intersections between nearby warped edges in the
                puzzle.
            self_intersecting_thresh: Float. Two points nearer than this value
                are considered intersecting.
            self_intersecting_stride: Int. Stride for sub-sampling the path when
                detecting self-intersections. This is useful only to speed up
                the self-intersection detection, which is the slowest part of
                the code.
        """
        self._curvature_object = curvature_object
        self._flattening_fn = flattening_fn
        self._self_intersecting_thresh = self_intersecting_thresh
        self._path_squishing_factor = path_squishing_factor
        self._self_intersecting_stride = self_intersecting_stride

        self._resolution = self._curvature_object.resolution

        # Compute matrix of path point index pairs to evaluate for
        # self-intersection
        linspace = np.linspace(
            0., 1., self._resolution // self._self_intersecting_stride)
        nearby = np.abs(linspace[np.newaxis] - linspace[:, np.newaxis])
        self._to_eval_for_self_intersection = (
            nearby > 1.5 * self._self_intersecting_thresh)

    def _self_intersecting(self, path):
        # Detect if curve comes close to self-intersecting
        path = path[::self._self_intersecting_stride]
        point_dists = np.linalg.norm(
            path[np.newaxis] - path[:, np.newaxis], axis=2)
        self_intersecting_points= self._to_eval_for_self_intersection * (
            point_dists < self._self_intersecting_thresh)
        if np.sum(self_intersecting_points):
            return True
        else:
            return False

    def __call__(self, start, end):
        # Sample and integrate curvature to get a warped path
        curvature = self._curvature_object() / self._resolution
        thetas = np.cumsum(curvature)
        vectors = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
        path = np.cumsum(vectors, axis=0)
        path_length = path.shape[0]

        # Linearly transform path so endpoints are [0, 0] and [0, 1]
        path -= path[0]
        path_end = path[-1]
        path /= np.linalg.norm(path_end)
        theta = np.arctan2(path_end[1], path_end[0])
        rotation_matrix = np.array([
            [np.cos(theta), -1. * np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        path = np.matmul(path, rotation_matrix)

        # Get straight path for weighted-average flattening
        straight_path = np.stack(
            [np.linspace(0., 1., path_length), np.zeros(path_length)], axis=1,
        )

        # Flatten endpoints by taking weighted average with straight_path
        mixing_weights = self._flattening_fn(path_length)[:, np.newaxis]
        path = mixing_weights * path + (1. - mixing_weights) * straight_path

        # Squish path along the y-axis
        path[:, 1] *= self._path_squishing_factor

        # Reject if curve comes close to self-intersecting
        if self._self_intersecting(path):
            return self(start, end)

        # Linearly transform path so endpoints are start and end
        start_end_vec = end - start
        theta = np.arctan2(start_end_vec[1], start_end_vec[0])
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],
            [-1. * np.sin(theta), np.cos(theta)]
        ])
        path = np.matmul(path, rotation_matrix)
        path *= np.linalg.norm(start_end_vec)
        path += start[np.newaxis]

        return path


class WarpEdges():
    """WarpEdges class.
    
    This is a transformation that warps the edges of a puzzle when called on it.
    Only the edges of the puzzle that are not along holes are warped. Warping an
    edge consists of replacing the edge with a warped path generated from a
    WarpedPath() instance, subject to rejection sampling if nearby puzzle edges
    intersect.
    """

    def __init__(self,  warped_path_object, intersection_thresh=0.04):
        """Constructor.
        
        Args:
            warped_path_object: Instance of WarpedPath.
            intersection_thresh: Float. Threshold distance. If two warped edges
                come within this distance of each other, they are considered
                intersecting and one of them is resampled.
        """
        logging.info('Constructing WarpEdges transform')

        self._warped_path_object = warped_path_object
        self._intersection_thresh = intersection_thresh

    def _intersects(self, path_0, path_1):
        """Detect whether two paths intersect.
        
        This intersection detection is the slowest part of the code. Consider
        alternative algorithms to improve runtime.
        """
        endpoint_buffer_0 = int(np.floor(0.1 * len(path_0)))
        endpoint_buffer_1 = int(np.floor(0.1 * len(path_1)))
        path_0 = path_0[endpoint_buffer_0: -endpoint_buffer_0]
        path_1 = path_1[endpoint_buffer_1: -endpoint_buffer_1]
        path_dists = np.linalg.norm(
            path_0[np.newaxis] - path_1[:, np.newaxis], axis=2)
        if np.sum(path_dists < self._intersection_thresh):
            return True
        else:
            return False
    
    def __call__(self, puzzle):
        """Apply transform to a puzzle."""
        logging.info('    Applying edge warping')

        start_time = time.time()
        mesh = puzzle.mesh

        # Identify which edges are parts of holes so we don't warp them
        all_hole_edges = []
        for hole in puzzle.holes:
            all_hole_edges.extend(hole['edges'])

        # Warp all the non-hole edges
        warped_edges = []
        for k, v in mesh.items():
            if k in all_hole_edges:
                continue
            start = v[0]
            end = v[-1]
            
            # Find edges that share a vertex
            neighboring_edges = []
            for edge in puzzle.edges:
                if k[0] in edge or k[1] in edge:
                    if edge == k:
                        continue
                    if edge in warped_edges:
                        neighboring_edges.append(edge)
            
            # Sample a warped edge that does not warped neighboring edges using
            # rejection sampling.
            success = False
            for _ in range(_MAX_TRIES):
                warped_edge = self._warped_path_object(start, end)
                intersects = False
                for edge in neighboring_edges:
                    other_edge_mesh = mesh[edge]
                    if self._intersects(warped_edge, other_edge_mesh):
                        intersects = True
                        break
                if intersects:
                    continue
                success = True
                break
            if not success:
                raise ValueError(
                    '_MAX_TRIES exceeded. This can happen if you get unlucky '
                    'with the random number generator. Try running the same '
                    'code again. If you keep getting this error, then you are '
                    'probably trying to warp edges too aggressively and should '
                    'adjust parameters to make edges more flat.'
                )

            warped_edges.append(k)
            mesh[k] = warped_edge
        
        # Report runtime of this function
        end_time = time.time()
        logging.info(f'    Edge warping runtime: {end_time - start_time}')
