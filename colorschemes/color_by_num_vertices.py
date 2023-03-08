"""Color polygons by number of vertices."""


class ColorByNumVertices():
    """Color polygons by number of vertices."""

    def __init__(self, num_vertices_to_rgb):
        """Constructor.
        
        Args:
            num_vertices_to_rgb: Function taking integer and return an RGB color
                for all polygons with that number of vertices.
        """
        self._num_vertices_to_rgb = num_vertices_to_rgb

    def __call__(self, puzzle):
        colors = {
            face: self._num_vertices_to_rgb(len(face)) for face in puzzle.faces
        }
        return colors
