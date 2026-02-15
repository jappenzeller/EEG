"""
DynamoDB session metadata for tracking experiments and results.

Table schema:
    PK: subject_id (String)
    SK: session_id (String)
    Attributes: session_type, duration_sec, n_channels, sample_rate,
                s3_uri, status, created_at, processing_results, etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from openbci_eeg.config import AWSConfig

logger = logging.getLogger(__name__)


def _get_table(config: Optional[AWSConfig] = None):
    """Get DynamoDB table resource."""
    if config is None:
        config = AWSConfig()
    dynamodb = boto3.resource("dynamodb", region_name=config.region)
    return dynamodb.Table(config.dynamodb_table)


def _sanitize_for_dynamo(obj: Any) -> Any:
    """Convert floats to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _sanitize_for_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_dynamo(v) for v in obj]
    return obj


def record_session_metadata(
    subject_id: str,
    session_id: str,
    session_type: str = "resting",
    duration_sec: float = 0.0,
    n_channels: int = 16,
    sample_rate: float = 125.0,
    s3_uri: str = "",
    extra: Optional[dict] = None,
    config: Optional[AWSConfig] = None,
) -> None:
    """
    Record session metadata to DynamoDB.

    Args:
        subject_id: Subject identifier (partition key).
        session_id: Session identifier (sort key).
        session_type: Type of recording session.
        duration_sec: Recording duration.
        n_channels: Number of EEG channels.
        sample_rate: Sample rate in Hz.
        s3_uri: S3 location of raw data.
        extra: Additional metadata fields.
        config: AWS config.
    """
    table = _get_table(config)

    item = {
        "subject_id": subject_id,
        "session_id": session_id,
        "session_type": session_type,
        "duration_sec": Decimal(str(duration_sec)),
        "n_channels": n_channels,
        "sample_rate": Decimal(str(sample_rate)),
        "s3_uri": s3_uri,
        "status": "recorded",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        item.update(_sanitize_for_dynamo(extra))

    try:
        table.put_item(Item=item)
        logger.info("Session metadata recorded: %s/%s", subject_id, session_id)
    except ClientError as e:
        logger.error("Failed to record metadata: %s", e)
        raise


def query_sessions(
    subject_id: str,
    session_type: Optional[str] = None,
    config: Optional[AWSConfig] = None,
) -> list[dict]:
    """
    Query sessions for a subject.

    Args:
        subject_id: Subject to query.
        session_type: Optional filter by session type.
        config: AWS config.

    Returns:
        List of session metadata dicts.
    """
    table = _get_table(config)

    try:
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("subject_id").eq(subject_id)
        )
        items = response.get("Items", [])

        if session_type:
            items = [i for i in items if i.get("session_type") == session_type]

        return items

    except ClientError as e:
        logger.error("Failed to query sessions: %s", e)
        raise
