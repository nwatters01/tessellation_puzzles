"""Demo warped edge generation.

This script runs warped edge generation and plots samples of curvature vectors,
noise vectors, flattening function, and warped paths. See edge_warping.py for
details of what these all mean.
"""

import numpy as np
from matplotlib import pyplot as plt
import edge_warping
import time

_PLOT_CURVATURES = False
_PLOT_NOISE = False
_PLOT_FLATTENING_FUNCTION = False
_PLOT_PATHS = True


def main():

    ############################################################################
    #### CREATE SAMPLING CLASSES
    ############################################################################

    sample_noise=edge_warping.get_sample_noise()
    curvature_object = edge_warping.Curvature(
        sample_curvature_magnitude=(
            edge_warping.get_sample_curvature_magnitude()),
        sample_curve_length=edge_warping.get_sample_curve_length(),
        sample_straight_length=edge_warping.get_sample_straight_length(),
        sample_noise=sample_noise,
    )
    flattening_fn = edge_warping.get_flattening_fn()
    warped_path_object = edge_warping.WarpedPath(
        curvature_object, flattening_fn)

    ############################################################################
    #### PROFILE CURVATURE AND PATH GENERATION
    ############################################################################
    
    # Report runtime of curvature sampling
    curvature_start_time = time.time()
    for _ in range(100):
        curvature_object()
    curvature_end_time = time.time()
    time_per_curvature_vector = (
        (curvature_end_time - curvature_start_time) / 100)
    print(f'Curvature runtime: {time_per_curvature_vector}')

    # Report runtime of path sampling
    path_start_time = time.time()
    for _ in range(100):
        warped_path_object(start=np.array([0., 0.]), end=np.array([1., 1.]))
    path_end_time = time.time()
    time_per_path = (path_end_time - path_start_time) / 100
    print(f'Warped path runtime: {time_per_path}')

    ############################################################################
    #### MAKE PLOTS
    ############################################################################
    
    if _PLOT_CURVATURES:
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for ax_row in axes:
            for ax in ax_row:
                curvature = curvature_object()
                ax.plot(curvature)
        fig.suptitle('Sampled curvatures')
        plt.tight_layout()

    if _PLOT_NOISE:
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for ax_row in axes:
            for ax in ax_row:
                noise = sample_noise(50)
                ax.plot(noise)
        fig.suptitle('Sampled noise')
        plt.tight_layout()

    if _PLOT_FLATTENING_FUNCTION:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(flattening_fn(50))
        plt.title('Flattening function')
        plt.tight_layout()
    
    if _PLOT_PATHS:
        fig, axes = plt.subplots(5, 5, figsize=(8, 8))
        for ax_row in axes:
            for ax in ax_row:
                path = warped_path_object(
                    start=np.array([0., 0.]), end=np.array([1., 1.]))
                ax.scatter(path[:, 0], path[:, 1], s=3)
                ax.set_aspect('equal', 'box')
                ax.axis('off')
        fig.suptitle('Sampled paths')
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()