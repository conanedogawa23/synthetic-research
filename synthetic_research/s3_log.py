"""Keep one JSON file per session in S3.

Each question is added to that session file. An improved turn replaces the
earlier record with the same turn id. The interview still completes if the
bucket is unset or the upload fails.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - the server venv installs boto3
    boto3 = None
    ClientError = Exception


def read_session_log(session_id: str) -> dict | None:
    bucket = os.environ.get("S3_LOG_BUCKET", "").strip()
    if not bucket or boto3 is None:
        return None
    key = f"sessions/{session_id}.json"
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1")
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        print(f"[s3-log] {type(error).__name__}: {error}", flush=True)
        return None
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return payload if isinstance(payload, dict) and payload else None


def log_question(record: dict) -> str | None:
    bucket = os.environ.get("S3_LOG_BUCKET", "").strip()
    if not bucket or boto3 is None:
        if bucket and boto3 is None:
            print("[s3-log] boto3 is not installed", flush=True)
        return None
    session_id = str(record.get("sessionId") or "session")
    key = f"sessions/{session_id}.json"
    client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1")
    try:
        current = _read(client, bucket, key)
        merged = _merge(current, record)
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(merged, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as error:  # noqa: BLE001
        print(f"[s3-log] {type(error).__name__}: {error}", flush=True)
        return None
    print(f"[s3-log] s3://{bucket}/{key}", flush=True)
    return key


def _read(client, bucket: str, key: str) -> dict:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return {}
        raise
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _merge(current: dict, record: dict) -> dict:
    questions = [item for item in current.get("questions") or [] if isinstance(item, dict)]
    turn_id = str(record.get("turnId") or "")
    entry = {key: value for key, value in record.items() if key not in {"sessionId", "campaignId"}}
    if turn_id:
        for index, item in enumerate(questions):
            if str(item.get("turnId") or "") == turn_id:
                questions[index] = entry
                break
        else:
            questions.append(entry)
    else:
        questions.append(entry)
    return {
        "sessionId": record.get("sessionId") or current.get("sessionId"),
        "campaignId": record.get("campaignId") or current.get("campaignId"),
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": questions,
    }
