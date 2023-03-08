"""Puzzle configuration."""

import numpy as np
from tessellations import degree_3_4_6_4
from transforms import smooth_holes
from transforms import radial_stretch
from transforms import dilate_holes
from transforms import edge_warping
from transforms import twist
import colorschemes as colorscheme_lib


def _num_vertices_to_rgb(n):
    if n == 3:
        return (1., 0., 0.)
    elif n == 4:
        return (0., 1., 0.)
    elif n == 6:
        return (0., 0., 1.)
    else:
        raise ValueError(f'Invalid number of vertices {n}')


def get_puzzle():
    # Initialize puzzle as a tessellation
    colorscheme = colorscheme_lib.ColorByNumVertices(_num_vertices_to_rgb)
    puzzle = degree_3_4_6_4.Degree3464(
        radius=8, mesh_interval=0.025, hex_fill_prob_tau=0.3,
        hex_fill_prob_baseline=-0.08, colorscheme=colorscheme)

    # Smooth holes
    smooth_holes.smooth_holes(puzzle)

    # Add edge warping
    curvature_object = edge_warping.Curvature(
        sample_curvature_magnitude=(
            edge_warping.get_sample_curvature_magnitude()),
        sample_curve_length=edge_warping.get_sample_curve_length(),
        sample_straight_length=edge_warping.get_sample_straight_length(),
        sample_noise=edge_warping.get_sample_noise(),
    )
    warped_path_object = edge_warping.WarpedPath(
        curvature_object,
        edge_warping.get_flattening_fn(),
    )
    edge_warping.WarpEdges(warped_path_object)(puzzle)

    # Dilate holes
    dilation_tau_fn = dilate_holes.normal_dilation_tau_fn(
        loc_intercept=-2., loc_slope=0.65, scale_slope=0.06,
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
