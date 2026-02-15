from importlib.metadata import version

__version__: str = version("ruranges")

from . import numpy

__all__ = ["numpy", "__version__"]
