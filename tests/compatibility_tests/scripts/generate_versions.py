import json

import semver
import subprocess
import sys

def generate_versions(to_version: str, num_minor_versions_to_test: int = 3) -> list:
    """
    Generate a list of previous versions based on the target version.

    This function generates a list of previous versions for a given target version.
    It includes all the previous patch versions of the same minor version if applicable,
    and the latest patch versions for preceding minor versions of up to num_minor_versions_to_test.

    Args:
        to_version (str): The target version to generate previous versions for.
        num_minor_versions_to_test (int): The number of previous minor versions to generate. Defaults to 3.

    Returns:
        list: A list of previous versions as strings.
    """
    target_version = semver.VersionInfo.parse(to_version)
    versions = []

    # If this is a patch release, add the previous patch version of the same minor version
    if target_version.patch > 0:
        versions.extend(
            f"{target_version.major}.{target_version.minor}.{i}"
            for i in range(target_version.patch - 1, -1, -1)
        )

    # Gather the latest patch version for each preceding minor version
    minor = target_version.minor - 1
    for _ in range(num_minor_versions_to_test):
        if minor < 0:
            break
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
