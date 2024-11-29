import json

import semver
import subprocess
import sys

def generate_versions(to_version: str, num_versions: int = 3) -> list:
    """
    Generate a list of previous versions based on the target version.

    This function generates a list of previous versions for a given target version.
    It includes the previous patch version of the same minor version if applicable,
    and the latest patch versions for preceding minor versions.

    Args:
        to_version (str): The target version to generate previous versions for.
        num_versions (int): The number of previous versions to generate. Defaults to 3.

    Returns:
        list: A list of previous versions as strings.
    """
    target_version = semver.VersionInfo.parse(to_version)
    versions = []

    # If this is a patch release, add the previous patch version of the same minor version
    if target_version.patch > 0:
        prev_patch_version = f"{target_version.major}.{target_version.minor}.{target_version.patch - 1}"
        versions.append(prev_patch_version)

    # Gather the latest patch version for each preceding minor version
    minor = target_version.minor - 1
    while len(versions) < num_versions and minor >= 0:
        # Get all tags for the given minor version, sort, and pick the latest patch
        tags = subprocess.check_output(
            ["git", "tag", "--list", f"{target_version.major}.{minor}.*"],
            text=True
        ).splitlines()

        # Filter and find the latest patch version tag
        if tags:
            latest_patch = max(tags, key=semver.VersionInfo.parse)
            versions.append(latest_patch.lstrip("v"))
        minor -= 1
    return versions

if __name__ == "__main__":
    to_version = sys.argv[1]  # Get to version from the command line
    num_versions = sys.argv[2] # Get number of versions to generate
    versions = generate_versions(to_version, int(num_versions))
    print(json.dumps(versions))  # Output versions as Json
