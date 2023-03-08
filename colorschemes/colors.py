"""List valid colors for colorschemes."""
from enum import Enum


class Color(Enum):
    """List valid colors for colorschemes."""
    PINK = (232/255, 30/255, 134/255)
    GREEN = (89/255, 237/255, 14/255)
    BLUE = (20/255, 137/255, 219/255)
    YELLOW = (245/255, 225/255, 5/255)
    ORANGE = (245/255, 121/255, 5/255)
    CLEAR = (255/255, 255/255, 255/255)
