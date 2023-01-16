"""Plot a puzzle."""

import importlib
import logging
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Point to a config in `configs/` directory
_CONFIG = '.degree_3_4_6_4_small_v0'


def _plot_puzzle(puzzle):
    """Plot puzzle."""
    logging.info('Plotting puzzle')

    _, ax = plt.subplots(figsize=(9, 9))

    # Render faces in gray
    logging.info('    Rendering faces')
    faces = puzzle.faces
    for f in faces:
        vertices = puzzle._face_to_mesh_polygon(f)
        polygon = patches.Polygon(vertices, closed=True)
        ax.add_collection(collections.PatchCollection([polygon], color='0.5'))

    # Plot edges in blue
    logging.info('    Rendering edges')
    flat_mesh = puzzle.flatten_mesh()
    plt.scatter(flat_mesh[:, 0], flat_mesh[:, 1], c='b', s=5)

    # Plot vertices in red
    logging.info('    Rendering vertices')
    vertices = puzzle.vertices
    plt.scatter(vertices[:, 0], vertices[:, 1], c='r', s=10)

    # Display plot
    logging.info('    Displaying figure')
    plt.show()


def main():
    config = importlib.import_module(_CONFIG, package='configs')
    puzzle = config.get_puzzle()
    _plot_puzzle(puzzle)


if __name__ == "__main__":
    main()
