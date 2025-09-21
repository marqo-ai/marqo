from pathlib import Path
from urllib.parse import urlparse

import botocore.exceptions
import fsspec
from tqdm import tqdm

from marqo_model_management_container.errors.common import ModelDownloadError


class TritonModelDownloader:
    """
    Build a Triton-compatible repo structure and download model files.

    Layout:
      <base_dir>/<model_name>/
        ├─ config.pbtxt   (optional)
        └─ 1/
           └─ *           (downloaded files)
    """

    def __init__(
        self,
        sources: list[str],
        base_dir: str,
        model_name: str,
        config_pbtxt: str | None = None,
        overwrite: bool = False,
    ):
        self.sources= sources
        self.base_dir = Path(base_dir)
        self.model_name = model_name
        self.config_pbtxt = config_pbtxt
        self.overwrite = overwrite

    def _version_dir(self) -> Path:
        root = (self.base_dir / self.model_name).resolve()
        (root / "1").mkdir(parents=True, exist_ok=True)
        if self.config_pbtxt is not None:
            (root / "config.pbtxt").write_text(self.config_pbtxt)
        return root / "1"

    @staticmethod
    def _basename_from_uri(uri: str) -> str:
        return Path(urlparse(uri).path).name or "model.onnx"

    def _download_with_progress(self, fs, path: str, dest: Path, chunk_size: int = 1024 * 1024):
        """Download a file with a tqdm progress bar."""
        try:
            info = fs.info(path)
        except botocore.exceptions.NoCredentialsError as e:
            raise ModelDownloadError(
                "Marqo cannot find your AWS credentials to download the model from S3. "
                "Please ensure your AWS credentials are configured correctly. You can mount "
                "your AWS credentials file into the container /root/.aws/credentials. Alternatively, "
                "you can provide the model files via a publicly accessible URL ") from e
        except FileNotFoundError as e:
            raise ModelDownloadError(f"The specified model file was not found: {path}. Please check "
                                     f"the provided source and ensure the container has access to it ") from e
        size = info.get("size", None)
        with fs.open(path, "rb") as fsrc, open(dest, "wb") as fdst, tqdm(
                total=size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar:
                for chunk in iter(lambda: fsrc.read(chunk_size), b""):
                    fdst.write(chunk)
                    bar.update(len(chunk))

    def prepare_and_download(self) -> list[Path]:
        version_dir = self._version_dir()
        srcs = self.sources

        out_paths: list[Path] = []
        for uri in srcs:
            fname = self._basename_from_uri(uri)
            dest = version_dir / fname

            if dest.exists() and not self.overwrite:
                out_paths.append(dest)
                continue

            fs, path = fsspec.core.url_to_fs(uri)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._download_with_progress(fs, path, dest)
            out_paths.append(dest)

        return out_paths
