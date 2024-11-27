import hashlib

from tqdm import tqdm
import os
from botocore.exceptions import ClientError

import boto3
from boto3.s3.transfer import TransferConfig


def md5_checksum(file_path):
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def parse_s3_url(s3_url):
    """
    Parse an S3 URL into bucket name and key prefix.

    Args:
        s3_url (str): The S3 URL in the format 's3://bucket-name/key-prefix/'.

    Returns:
        tuple: (bucket_name, key_prefix)
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL. Must start with 's3://'.")
    s3_url = s3_url.replace("s3://", "", 1)
    parts = s3_url.split("/", 1)
    bucket_name = parts[0]
    key_prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, key_prefix


class ProgressPercentage:
    def __init__(self, file_name, total_size):
        self.file_name = file_name
        self.total_size = total_size
        self.progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name)

    def __call__(self, bytes_amount):
        self.progress_bar.update(bytes_amount)

    def close(self):
        self.progress_bar.close()


def download_file_with_progress(s3_client, bucket_name, s3_key, local_file_path):
    """
    Download a file from S3 with a progress bar.

    Args:
        s3_client: Boto3 S3 client instance.
        bucket_name (str): S3 bucket name.
        s3_key (str): S3 key of the file to download.
        local_file_path (str): Local path to save the downloaded file.
    """
    # Get the file size from S3
    response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
    file_size = response['ContentLength']

    # Progress bar callback
    progress = ProgressPercentage(s3_key, file_size)

    # Transfer configuration for efficient downloading
    config = TransferConfig(multipart_threshold=8 * 1024 * 1024, max_concurrency=10)

    try:
        s3_client.download_file(
            Bucket=bucket_name,
            Key=s3_key,
            Filename=local_file_path,
            Config=config,
            Callback=progress,
        )
    finally:
        progress.close()


def download_s3_directory(s3_url, local_dir):
    """
    Download an S3 directory to a local directory with integrity check and recovery.

    Args:
        s3_url (str): The S3 URL in the format 's3://bucket-name/key-prefix/'.
        local_dir (str): Local directory to save files.

    Returns:
        str: Path to the target directory where files are downloaded.
    """
    bucket_name, s3_prefix = parse_s3_url(s3_url)

    # Use the S3 prefix as the target directory name
    target_dir_name = os.path.basename(s3_prefix.rstrip('/'))
    target_dir = os.path.join(local_dir, target_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    for page in pages:
        if 'Contents' not in page:
            return target_dir

        for obj in page['Contents']:
            s3_key = obj['Key']
            relative_path = s3_key[len(s3_prefix):]
            local_file_path = os.path.join(target_dir, relative_path)

            # Create local directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Skip directories
            if s3_key.endswith('/'):
                continue

            try:
                # Check for file existence
                if os.path.exists(local_file_path):
                    local_size = os.path.getsize(local_file_path)
                    remote_etag = obj['ETag'].strip('"')
                    remote_size = obj['Size']

                    # Skip redownload if file sizes match (for multipart files)
                    if local_size == remote_size:
                        continue

                    # Perform checksum validation only for non-multipart files
                    if "-" not in remote_etag:
                        local_md5 = md5_checksum(local_file_path)
                        if local_md5 == remote_etag:
                            continue

                # Download the file with progress bar
                download_file_with_progress(s3, bucket_name, s3_key, local_file_path)

                # Verify checksum after download for non-multipart files
                remote_etag = obj['ETag'].strip('"')
                if "-" not in remote_etag:
                    local_md5 = md5_checksum(local_file_path)
                    if local_md5 != remote_etag:
                        os.remove(local_file_path)
                        download_file_with_progress(s3, bucket_name, s3_key, local_file_path)

            except ClientError:
                pass

    return target_dir
