import os
import pathlib

PROJECT_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent

SAVE_DIR = os.path.join(PROJECT_DIR, 'saved')