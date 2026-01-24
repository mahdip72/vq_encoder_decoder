import importlib.metadata

from graphein import verbose

# Disable graphein import warnings
verbose(False)


try:
    __version__ = importlib.metadata.version("gcpnet")
except importlib.metadata.PackageNotFoundError:  # fall back when running from cloned sources
    __version__ = "0.0.0"
