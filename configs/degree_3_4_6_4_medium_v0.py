"""Puzzle configuration."""

import numpy as np
from tessellations import degree_3_4_6_4
from transforms import smooth_holes
from transforms import radial_stretch
from transforms import dilate_holes
from transforms import twist


def get_puzzle():
    # Initialize puzzle as a tessellation
    puzzle = degree_3_4_6_4.Degree3464(
        radius=25, mesh_interval=0.1, hex_fill_prob_tau=0.025,
        hex_fill_prob_baseline=-0.03)

    # Smooth holes
    smooth_holes.smooth_holes(puzzle)

    # Dilate holes
    dilation_tau_fn = dilate_holes.normal_dilation_tau_fn(
        loc_intercept=-1., loc_slope=0.2, scale_slope=0.06,
        min_dilation_tau=-1.)
    dilate_holes.DilateHoles(dilation_tau_fn, puzzle)(puzzle)

    # Add random twists to space
    twist_transform = twist.Twist(
        puzzle.get_bound(),
        density=0.1,
        rotation_magnitude=0.1 * np.pi,
        basin_tau_baseline=0.05,
        basin_tau_variance=0.05,
    )
    twist_transform(puzzle)

    # Radially stretch puzzle
    radial_stretch.RadialStretch(tau=0.01)(puzzle)

    return puzzle
