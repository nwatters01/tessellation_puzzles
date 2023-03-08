"""Color polygons by k means clusters."""

import numpy as np
from sklearn.cluster import KMeans
from itertools import cycle

from colorschemes.colors import Color


def _get_centroid(face, puzzle):
    vertices = [puzzle.vertices[v] for v in face]
    centroid = np.mean(vertices, axis=0)
    return centroid


class ColorByKMeans():
    """Color polygons by k means clusters."""

    def __init__(self, k):
        """Constructor.

        Args:
            k: Number of clusters for k means.
        """
        self._k = k

    def __call__(self, puzzle):
        centroids = [_get_centroid(face, puzzle) for face in puzzle.faces]
        kmeans = KMeans(n_clusters=self._k).fit(centroids)
        cluster_labels = kmeans.labels_
        cluster_label_to_color = dict(zip(set(kmeans.labels_), cycle(Color)))

        colors = {
            face: cluster_label_to_color[cluster_label].value for face, cluster_label in zip(puzzle.faces, cluster_labels)
        }
        return colors
