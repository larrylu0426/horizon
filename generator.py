import os
import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


current_dir = Path()
assert len(sys.argv) == 2, \
    'Specify the project path. Example: python generator.py ~/my-project'

path = sys.argv[1].split("/")
if len(path) == 1:
    project_name = path[0]
    target_dir = Path(os.getcwd()) / project_name
else:
    target_dir = Path(sys.argv[1])

ignore = [".vscode", ".git", "data", "generator.py", "LICENSE",
          "wandb", "__pycache__", "Readme.md"]
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore))
print('New project initialized at: ', target_dir)
