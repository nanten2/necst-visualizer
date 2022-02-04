try:
    from importlib_metadata import version
except ImportError:
    from importlib.metadata import version  # Python 3.8+

try:
    __version__ = version("necst_visualizer")
except:
    __version__ = None

from .scan_check import ScanCheck, VisualizeScan
