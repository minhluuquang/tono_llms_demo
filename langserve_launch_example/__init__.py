"""langserve_launch_example package."""
from importlib import metadata

from langserve_launch_example.chain import get_chain
from dotenv import load_dotenv

load_dotenv()

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [__version__, "get_chain"]
