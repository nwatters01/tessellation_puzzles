"""Plot a puzzle."""

import importlib
import logging
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
import scipy.spatial
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Point to a config in `configs/` directory
_CONFIG = '.degree_3_4_6_4_tiny_v0'

# Whether to shade puzzle faces. This is pretty time-consuming in its current
# implementation, so for speed you may want to turn it off.
_RENDER_FACES = True


def _plot_puzzle(puzzle):
    """Plot puzzle."""
    logging.info('Plotting puzzle')

    _, ax = plt.subplots(figsize=(9, 9))
    
    # Render faces in gray, if necessary
    if _RENDER_FACES:
        logging.info('    Rendering faces')
        faces = puzzle.faces
        for f in faces:
            vertices = puzzle._face_to_mesh_polygon(f, stride=5)
            polygon = patches.Polygon(vertices, closed=True)
            ax.add_collection(
                collections.PatchCollection([polygon], color=puzzle.colors[f]))

    # Render edges in pink
    logging.info('    Rendering edges')
    for edge_points in puzzle.mesh.values():
        ax.plot(edge_points[:, 0], edge_points[:, 1], c='pink')

    # Render vertices in red
    logging.info('    Rendering vertices')
    vertices = puzzle.vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], c='r', s=5)

    # Display plot
    logging.info('    Displaying figure')

    plt.show()


def main():
    config = importlib.import_module(_CONFIG, package='configs')
    puzzle = config.get_puzzle()
    _plot_puzzle(puzzle)


if __name__ == "__main__":
    main()
