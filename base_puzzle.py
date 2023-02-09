"""Base puzzle class with common functions."""

import abc
import logging
import numpy as np


class BasePuzzle(abc.ABC):
    """Base puzzle class."""

    def __init__(self, mesh_interval=0.1):
        """Constructor.
        
        Args:
            mesh_interval: Scalar. Bin size for edge discretization, in units of
                polygon edge length. Should be less than 1. Smaller means
                smoother edges.
        """
        logging.info('Constructing base puzzle')

        self._mesh_interval = mesh_interval

        logging.info('    Setting up vertices')
        self._vertices = self._setup_vertices()
        logging.info('    Setting up edges')
        self._edges = self._setup_edges()
        logging.info('    Setting up faces')
        self._faces = self._setup_faces()
        logging.info('    Setting up mesh')
        self._mesh = self._setup_mesh()
        logging.info('    Setting up holes')
        self._holes = self._setup_holes()

    @abc.abstractmethod
    def _setup_vertices(self):
        """Create tessellation vertices.
        
        Returns:
            [num_vertices, 2] float array of vertices coordinates.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_edges(self):
        """Create tessellation edges.
        
        Returns:
            List of integer 2-tuples. Each element (i, j) represents indices of
                vertices in self.vertices that are connected by an edge. To
                avoid redundancy, must satisfy i < j.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_faces(self):
        """Create tessellation faces.
        
        Returns:
            List of tuples of integers. Each tuple contains indices of vertices
                of a face, ordered either clockwise or counterclockwise for
                rendering to work properly.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_hole_faces(self):
        """Create tessellation hole faces.

        Hole faces are faces that are ommitted from the puzzle, left as holes.
        
        Returns:
            List of tuples of integers. Each tuple contains indices of vertices
                of a face, and must be an element of self.faces.
        """
        raise NotImplementedError

    def _setup_mesh(self):
        """Create a mesh by making a line of points along each edges.
        
        Each edge in the tessellation will be transformed into a curve in the
        puzzle. To track the transformations, each edge is represented as a line
        of points. This function creates those lines of points.

        Returns:
            mesh: Dictionary. Keys are pairs of vertex indices (i, j)
                corresponding to an edge in the tessellation. Values are [_, 2]
                float arrays of points dotting the correspoinding edge.
        """

        def discretize_line(start, end):
            dist = np.linalg.norm(end - start)
            num_points = int(np.ceil(dist / self._mesh_interval))
            x = np.linspace(start[0], end[0], num_points)
            y = np.linspace(start[1], end[1], num_points)
            return np.stack([x, y], axis=1)
        
        vertices = self.vertices
        edges = self.edges
        mesh = {}
        for (i, j) in edges:
            v_0 = vertices[i]
            v_1 = vertices[j]
            mesh[(i, j)] = discretize_line(v_0, v_1)
        
        return mesh

    def _setup_holes(self):
        """Create holes.
        
        Returns:
            holes: List of dictionaries. Each element has fields:
                'center': Size (2,) float array. Coordinates of center.
                'edges': List of integer 2-tuples containing edges of hole.
        """

        hole_faces = self._setup_hole_faces()
        holes = []
        for h in hole_faces:
            vertices = np.array([self.vertices[i] for i in h])
            center = np.mean(vertices, axis=0)
            hole_edges = []
            for i in range(len(h)):
                v_0 = h[i]
                v_1 = h[(i + 1) % len(h)]
                if v_0 < v_1:
                    hole_edges.append((v_0, v_1))
                else:
                    hole_edges.append((v_1, v_0))
            holes.append({'center': center, 'edges': hole_edges})
            self._remove_hole_from_faces(h)

        return holes

    def flatten_mesh(self):
        """Flatten self.mesh into a single array of edgepoints for plotting."""
        flat_mesh = []
        for v in self.mesh.values():
            flat_mesh.extend(v.tolist())
        return np.array(flat_mesh)

    def get_bound(self):
        """Get radial bound for entire puzzle."""
        flat_mesh = self.flatten_mesh()
        radii = np.linalg.norm(flat_mesh, axis=1)
        return np.max(radii)

    def _remove_hole_from_faces(self, hole):
        """Pop hole from self.faces."""
        pop_ind = np.argwhere([x == hole for x in self.faces])[0, 0]
        self.faces.pop(pop_ind)

    def _order_face_corners(self, corners):
        """Reorder corners to be counterclockwise."""
        vertices = np.array([self.vertices[i] for i in corners])
        center = np.mean(vertices, axis=0)
        relative_vertices = vertices - center
        angles = np.arctan2(relative_vertices[:, 0], relative_vertices[:, 1])
        ordering = np.argsort(angles)
        corners = tuple([corners[i] for i in ordering])
        return corners

    def _face_to_mesh_polygon(self, face, stride=1):
        """Convert face to a polygon with mesh edges.
        
        For rendering, we want to fill faces of the puzzle. This requires
        converting the face into a patch, namely a big polygon with mesh edges.
        This function creates that polygon.
        
        Args:
            face: Tuple of integer vertex indices.
            stride: Int. Stride for sub-sampling edge points. Get most accurate
                polygon mesh with stride 1, but rendering faces can be slow with
                high resolution, so you may want higher stride for speed of
                rendering.

        Returns:
            polygon: Size [_, 2] float array of polygon points outlining the
                face.
        """
        polygon = []
        for i in range(len(face)):
            v_0 = face[i]
            v_1 = face[(i + 1) % len(face)]
            if v_0 < v_1:
                edge = (v_0, v_1)
                correct_orientation = True
            else:
                edge = (v_1, v_0)
                correct_orientation = False
            
            edge_points = self.mesh[edge]
            if not correct_orientation:
                edge_points = edge_points[::-1]
            
            polygon.append(edge_points[::stride])
        
        polygon = np.concatenate(polygon, axis=0)
        return polygon

    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vertices

    @property
    def edges(self):
        return self._edges

    @property
    def faces(self):
        return self._faces

    @property
    def mesh(self):
        return self._mesh

    @property
    def holes(self):
        return self._holes