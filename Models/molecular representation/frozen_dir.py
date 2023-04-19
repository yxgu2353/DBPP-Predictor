from os.path import dirname
import sys

def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return dirname(sys.executable)
    return dirname(__file__)