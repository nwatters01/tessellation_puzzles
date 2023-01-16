"""Smooth holes function."""

import logging
import numpy as np


def smooth_holes(puzzle):
    """Smooth holes in puzzle so they are round and have no sharp corners."""
    logging.info('Smoothing holes')
    
    for h in puzzle.holes:
        hole_center = h['center']
        for (i, j) in h['edges']:
            # Edge is in hole, so make it an arc
            v_0 = puzzle.vertices[i]
            v_1 = puzzle.vertices[j]
            v_0_dist = np.linalg.norm(v_0 - hole_center)
            v_1_dist = np.linalg.norm(v_1 - hole_center)
            if not np.isclose(v_0_dist, v_1_dist, rtol=0.001):
                raise ValueError("Hole is not in center of face.")
            edgepoints = puzzle.mesh[(i, j)]
            new_edgepoints = (
                hole_center + (edgepoints - hole_center) / np.linalg.norm(
                    edgepoints - hole_center, axis=1, keepdims=True)
            )
            puzzle.mesh[(i, j)] = new_edgepoints
