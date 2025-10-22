from pathlib import Path
import os

root_version_file_path = os.path.join(
    Path(__file__).resolve().parents[4], "version"
)  # The root 'version' file
docker_version_file_path = os.path.join(
    Path(__file__).resolve().parent, "version"
)  # The docker 'version' file

if os.path.exists(docker_version_file_path):
    version_file_path = docker_version_file_path
elif os.path.exists(root_version_file_path):
    version_file_path = root_version_file_path
else:
    raise FileNotFoundError(
        "Version file not found in either docker or root directory."
    )

with open(version_file_path) as f:
    __version__ = f.read().strip()


def get_version() -> str:
    return f"{__version__}"
