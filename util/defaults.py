import os

def get_default_path():
    """Get the absolute path to the repository directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))