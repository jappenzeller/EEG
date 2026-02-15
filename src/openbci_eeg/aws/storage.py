"""
S3 storage for raw EEG recordings, processed data, and results.

Bucket structure:
    s3://{bucket}/
    ├── raw/{subject_id}/{session_id}/
    │   ├── raw_data.npy
    │   └── metadata.json
    ├── processed/{subject_id}/{session_id}/
    │   ├── preprocessed.npy
    │   ├── pn_parameters.npz
    │   └── processing_log.json
    └── results/{subject_id}/{session_id}/
        ├── quantum_states.npz
        ├── fidelity_scores.csv
        └── predictions.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from openbci_eeg.config import AWSConfig

logger = logging.getLogger(__name__)


def _get_client(config: Optional[AWSConfig] = None):
    """Get boto3 S3 client."""
    if config is None:
        config = AWSConfig()
    return boto3.client("s3", region_name=config.region)


def upload_session(
    local_dir: str | Path,
    subject_id: str,
    session_id: str,
    prefix: str = "raw",
    config: Optional[AWSConfig] = None,
) -> str:
    """
    Upload a recording session directory to S3.

    Args:
        local_dir: Local directory containing session files.
        subject_id: Subject identifier.
        session_id: Session identifier.
        prefix: S3 prefix (raw, processed, or results).
        config: AWS config.

    Returns:
        S3 URI of the uploaded session.
    """
    if config is None:
        config = AWSConfig()

    client = _get_client(config)
    local_dir = Path(local_dir)
    s3_prefix = f"{prefix}/{subject_id}/{session_id}"

    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(local_dir)}"
            try:
                client.upload_file(str(file_path), config.bucket, s3_key)
                logger.debug("Uploaded %s → s3://%s/%s", file_path, config.bucket, s3_key)
            except ClientError as e:
                logger.error("Failed to upload %s: %s", file_path, e)
                raise

    s3_uri = f"s3://{config.bucket}/{s3_prefix}"
    logger.info("Session uploaded to %s", s3_uri)
    return s3_uri


def download_session(
    subject_id: str,
    session_id: str,
    local_dir: str | Path,
    prefix: str = "raw",
    config: Optional[AWSConfig] = None,
) -> Path:
    """
    Download a session from S3 to local directory.

    Args:
        subject_id: Subject identifier.
        session_id: Session identifier.
        local_dir: Local download directory.
        prefix: S3 prefix.
        config: AWS config.

    Returns:
        Path to local directory.
    """
    if config is None:
        config = AWSConfig()

    client = _get_client(config)
    local_dir = Path(local_dir) / session_id
    local_dir.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"{prefix}/{subject_id}/{session_id}/"

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=config.bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(s3_prefix):]
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(config.bucket, key, str(local_path))
            logger.debug("Downloaded %s", relative)

    logger.info("Session downloaded to %s", local_dir)
    return local_dir


def list_sessions(
    subject_id: Optional[str] = None,
    prefix: str = "raw",
    config: Optional[AWSConfig] = None,
) -> list[dict[str, str]]:
    """
    List available sessions in S3.

    Args:
        subject_id: Filter by subject. Lists all if None.
        prefix: S3 prefix to search.
        config: AWS config.

    Returns:
        List of dicts with 'subject_id' and 'session_id' keys.
    """
    if config is None:
        config = AWSConfig()

    client = _get_client(config)
    search_prefix = f"{prefix}/"
    if subject_id:
        search_prefix = f"{prefix}/{subject_id}/"

    sessions = []
    paginator = client.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(
            Bucket=config.bucket, Prefix=search_prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                parts = cp["Prefix"].rstrip("/").split("/")
                if len(parts) >= 3:
                    sessions.append({
                        "subject_id": parts[1],
                        "session_id": parts[2],
                    })
    except ClientError as e:
        logger.error("Failed to list sessions: %s", e)
        raise

    return sessions
