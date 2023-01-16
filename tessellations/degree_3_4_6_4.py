"""Tessellation with [triangle, square, triangle, hexagon] vertices."""

import base_puzzle
import logging
import numpy as np

_HALF_ROOT_3 = 0.5 * np.sqrt(3)


class Degree3464(base_puzzle.BasePuzzle):
    """Degree3464 tessellation class.
    
    This tessellation has around each vertex polygons
    [triangle, square, triangle, hexagon], in that order. All polygons are
    regular and all edges have the same length.
    """

    def __init__(self,
                 radius,
                 hex_fill_prob_tau=0.,
                 hex_fill_prob_baseline=0.,
                 mesh_interval=0.1):
        """Constructor.
        
        Args:
            radius: Scalar. Radius of the tesselation, in units of a polygon
                edge length.
            hex_fill_prob_tau: Scalar. Exponential decay of the fill probability
                of hexagons in the tessellation. Fill probability decays to zero
                as a function of radius.
            hex_fill_prob_baseline: Scalar. Probability added to the exponential
                fill probability. Typically negative, to ensure that all
                hexagons beyond a certain radius are not filled.
            mesh_interval: Scalar. Bin size for edge discretization, in units of
                polygon edge length. Should be less than 1. Smaller means
                smoother edges.
        """
        logging.info('Constructing Degree3464 tessellation')

        self._radius = radius
        self._hex_fill_prob_tau = hex_fill_prob_tau
        self._hex_fill_prob_baseline = hex_fill_prob_baseline

        logging.info('    Setting up hexagon centers')
        self._hexagon_centers = self._setup_hexagon_centers()

        super(Degree3464, self).__init__(mesh_interval=mesh_interval)
        
    def _setup_hexagon_centers(self):
        """Get centers of the hexagons in the tessellation."""
        # Make a large square lattice
        axis = np.arange(-self._radius, self._radius)
        x, y = np.meshgrid(axis, axis)
        vertices = np.stack([np.ravel(x), np.ravel(y)], axis=1)
        
        # Shear the square lattice to make it a triangular lattice
        shear_matrix = np.array([[1, 0], [0.5, _HALF_ROOT_3]])
        vertices = np.dot(vertices, shear_matrix)

        # Scale up the lattice so that edges in the final tessellation will have
        # length 1
        vertices =  (1 + 2 * _HALF_ROOT_3) * vertices

        # Remove vertices outside radius
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices[norms < self._radius]

        return vertices

    def _setup_vertices(self):
        """Create tessellation vertices.
        
        Returns:
            [num_vertices, 2] float array of vertices coordinates.
        """
        # Compute vertices of a zero-centered hexagon with unit sidelength
        hexagon_vertices = np.array([
            [0, 1],
            [_HALF_ROOT_3, 0.5],
            [_HALF_ROOT_3, -0.5],
            [0, -1],
            [-_HALF_ROOT_3, -0.5],
            [-_HALF_ROOT_3, 0.5],
        ])
        
        # Iterate through all hexagon centers, computing the vertices of each
        # hexagon. These are the vertices of the tessellation.
        self._vertices_per_hexagon = {}
        vertices = []
        for center in self._hexagon_centers:
            self._vertices_per_hexagon[tuple(center)] = tuple(range(
                len(vertices), len(vertices) + 6))
            vertices.extend((center + hexagon_vertices).tolist())

        return np.array(vertices)

    def _setup_edges(self):
        """Create tessellation edges.
        
        Returns:
            List of integer 2-tuples. Each element (i, j) represents indices of
                vertices in self.vertices that are connected by an edge. To
                avoid redundancy, i < j.
        """
        # Compute pairwise inter-vertex distance
        pairwise_diffs = (
            self.vertices[np.newaxis] - self.vertices[:, np.newaxis]
        )
        pairwise_dists = np.linalg.norm(pairwise_diffs, axis=2)

        # Find all unique vertex pairs that are distance ~1 apart. These are the
        # edges.
        connected = np.isclose(pairwise_dists, 1., rtol=0.01)
        edges = np.argwhere(connected)
        edges = edges[edges[:, 1] > edges[:, 0]]
        edges = [tuple(x) for x in edges]

        return edges

    def _setup_faces(self):
        """Create tessellation faces.
        
        Returns:
            List of tuples of integers. Each tuple contains indices of vertices
                of a face, ordered either clockwise or counterclockwise.
        """
        # Get hexagonal faces
        faces = list(self._vertices_per_hexagon.values())

        # Get square faces
        adjacent_hexagons = {}
        for i, c_0 in enumerate(self._hexagon_centers):
            for j, c_1 in enumerate(self._hexagon_centers):
                if j <= i:
                    continue
                adjacent = np.isclose(
                    np.linalg.norm(c_1 - c_0), (1 + 2 * _HALF_ROOT_3))
                if not adjacent:
                    continue
                vertices_inds_0 = self._vertices_per_hexagon[tuple(c_0)]
                vertices_0 = np.array([
                    self.vertices[i] for i in vertices_inds_0])
                vertices_inds_1 = self._vertices_per_hexagon[tuple(c_1)]
                vertices_1 = np.array([
                    self.vertices[i] for i in vertices_inds_1])
                dists = np.linalg.norm(
                    vertices_0[:, np.newaxis] - vertices_1[np.newaxis], axis=2)
                adjacent_verts_0 = np.argwhere(
                    np.isclose(np.min(dists, axis=1), 1.))[:, 0]
                adjacent_verts_1 = np.argwhere(
                    np.isclose(np.min(dists, axis=0), 1.))[:, 0]
                adjacent_verts_0 = [
                    vertices_inds_0[x] for x in adjacent_verts_0]
                adjacent_verts_1 = [
                    vertices_inds_1[x] for x in adjacent_verts_1]
                adjacent_hexagons[(i, j)] = (adjacent_verts_0, adjacent_verts_1)
                square = list(adjacent_verts_0) + list(adjacent_verts_1)
                faces.append(self._order_face_corners(square))

        # Get triangular faces
        for (i, j) in adjacent_hexagons.keys():
            for (l, k) in adjacent_hexagons.keys():
                if l != j:
                    continue
                if (i, k) not in adjacent_hexagons:
                    continue
                (verts_ij, verts_ji) = adjacent_hexagons[(i, j)]
                (verts_ik, verts_ki) = adjacent_hexagons[(i, k)]
                (verts_jk, verts_kj) = adjacent_hexagons[(j, k)]
                v_i = [x for x in verts_ij if x in verts_ik][0]
                v_j = [x for x in verts_ji if x in verts_jk][0]
                v_k = [x for x in verts_ki if x in verts_kj][0]
                faces.append((v_i, v_j, v_k))

        return faces

    def _setup_hole_faces(self):
        """Create tessellation hole faces.

        Hole faces are faces that are ommitted from the puzzle, left as holes.
        
        Returns:
            List of tuples of integers. Each tuple contains indices of vertices
                of a face, and is an element of self.faces.
        """

        def _prob_hole(face):
            """Probability of given face being a hole."""
            vertices = np.array([self.vertices[i] for i in face])
            center = np.mean(vertices, axis=0)
            r = np.linalg.norm(center)
            fill_prob = self._hex_fill_prob_baseline + (
                np.exp(-self._hex_fill_prob_tau * r))
            hole_prob = 1 - fill_prob
            return hole_prob

        hole_faces = []
        for face in self.faces:
            if len(face) != 6:  # Only hexagonal faces can be holes
                continue
            if np.random.rand() < _prob_hole(face):
                hole_faces.append(face)

        return hole_faces
