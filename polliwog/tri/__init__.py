from . import functions as _functions
from .functions import *
from .quad_faces import quads_to_tris

__all__ = _functions.__all__ + ["quads_to_tris"]
