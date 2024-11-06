import semver
import subprocess
import sys

def generate_versions(to_version: str, num_versions: int = 4) -> list:
    target_version = semver.VersionInfo.parse(to_version)
    versions = []

    # If this is a patch release, add the previous patch version of the same minor version
    if target_version.patch > 0:
        prev_patch_version = f"{target_version.major}.{target_version.minor}.{target_version.patch - 1}"
        versions.append(prev_patch_version)
        num_versions-=1

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
    print(" ".join(versions))  # Output versions as a comma-separated string
