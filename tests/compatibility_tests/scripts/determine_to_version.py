import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))
from marqo.version import __version__
import semver

def determine_to_version(run_commit_hash: str, current_marqo_version: str):
    """
    This function determines the to_version.
    It does so by looking at version.py file. However there can be times where active development is going on and
    Marqo developers have not yet updated the version.py file with the next (i.e To be released) version. In such cases, we need to
    determine the to version by looking at the latest tag in the git repository.

    If for version v, tag t exists, then it could mean that version v is already released and the developers are working on version v+1.
    We determine if this is the case by comparing commit hash of tag t and commit hash of the github workflow run. If they're different, we can
    conclude that version v is already released, and to test backwards compatibility we need to test against version v as well. Thus we set to_version = v+1.

    If the commit hash of tag t and commit hash of the github workflow run are the same, then we can conclude that this may be a re-run. Similar to this case,
    if the tag t for version v doesn't exist yet, we can determine that version v is the upcoming To be released version. In this case we set to_version = v.
    """
    tag = subprocess.check_output(["git", "tag", "--list", f"{current_marqo_version}"],
                                   text=True).splitlines() #Determine if tags exist for current_marqo_version picked from version.py file
    if tag: #If tag already exists for the current_marqo_version, it means that this version is already released and we are working towards the next version release, thus we need to treat this commit as commit of the next version release.
        try:
            tag_commit_hash = subprocess.check_output( #Determining commit hash of the tag
                ["git", "rev-list", "-n", "1", tag[0]],
                text=True
            ).strip()
            if tag_commit_hash != run_commit_hash: #If commit hashes don't match, it means that this commit is for the next version, thus we need to set to_version to version.bump_patch().
                to_version = semver.VersionInfo.parse(current_marqo_version).bump_patch()
                return str(to_version)
            elif tag_commit_hash == run_commit_hash: #If the commit hashes are the same - it means that this could be a manual re-run, in that case no need to set to_version to version.bump_patch().
                return current_marqo_version
        except subprocess.CalledProcessError as e:
            print(f"Error while determining to_version: {e}")
    else: #If tags don't exist, it means that this commit is for a new version whose tag is yet to be released, thus our to_version can be the version picked up from versions.py
        return current_marqo_version

if __name__ == "__main__":
    commit_hash = sys.argv[1]  # Get to version from the command line
    current_marqo_version = __version__
    to_version = determine_to_version(commit_hash, current_marqo_version)
    print(to_version)  # Output versions as a comma-separated string
