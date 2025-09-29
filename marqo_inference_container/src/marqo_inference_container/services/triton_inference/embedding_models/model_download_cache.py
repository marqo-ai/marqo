import os


def _get_project_root() -> str:
    project_root =  os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
    print(f"Project root determined as: {project_root}")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))


class ModelDownloadCache:
    """
    A class to manage cache paths for different model types.
    """
    open_clip_cache_path = os.path.join(_get_project_root(), '.cache/open_clip_cache/')
    # The hf_cache_path is managed by the hf_hub_download function
    hf_cache_path = os.path.join(_get_project_root(), '.cache/marqo_hf_cache/')

