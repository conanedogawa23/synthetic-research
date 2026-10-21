"""Chat completions for the respondent, follow-up, and insight pack.

Gemma on Bedrock is called with the instance role. A non-Bedrock chat URL
uses a bearer token.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from synthetic_research.model_http import read_json_response
from synthetic_research.prompts import (
    FOLLOW_UP_SYSTEM,
    PACK_SYSTEM,
    respondent_messages,
)

RESPONDENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "respondent_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "abstain": {"type": "boolean"},
                "reply": {"type": "string"},
                "evidenceIds": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["abstain", "reply", "evidenceIds"],
        },
    },
}


def parse_model_json(text: str) -> dict:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    body = fenced.group(1).strip() if fenced else raw
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(body[start : end + 1])


class ChatClient:
    def __init__(self, config: dict):
        self._config = config

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        messages: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict:
        content = self._chat(
            model,
            system,
            user,
            max_tokens,
            messages=messages,
            response_format=response_format,
        )
        try:
            return parse_model_json(content)
        except (ValueError, json.JSONDecodeError):
            repaired = self._chat(
                model,
                "Return one JSON object and nothing else.",
                f"Convert this into one JSON object, keeping the same fields:\n{content}",
                max_tokens,
            )
            return parse_model_json(repaired)

    def respondent(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["fast_model"],
            system="",
            user="",
            max_tokens=4096,
            messages=respondent_messages(user),
            response_format=RESPONDENT_RESPONSE_FORMAT,
        )

    def pack(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["reasoning_model"],
            system=PACK_SYSTEM,
            user=user,
            max_tokens=4096,
        )

    def follow_up(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["fast_model"],
            system=FOLLOW_UP_SYSTEM,
            user=user,
            max_tokens=120,
        )

    def _chat(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        *,
        messages: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> str:
        chat_messages = messages or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            payload = self._post_chat(_chat_body(model, chat_messages, max_tokens, response_format))
        except RuntimeError as error:
            if response_format is None or " 400:" not in str(error):
                raise
            print("[chat] rejected response_format, retrying without it", flush=True)
            payload = self._post_chat(_chat_body(model, chat_messages, max_tokens, None))
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("The chat model returned an empty completion.")
        return content

    def _post_chat(self, body: bytes) -> dict:
        chat_url = f"{self._config['base_url']}/chat/completions"
        if _is_bedrock(self._config["base_url"]):
            return read_json_response(lambda: _signed_bedrock_request(chat_url, body), "chat", "Bedrock")
        return read_json_response(
            lambda: urllib.request.Request(
                chat_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            ),
            "chat",
            "Chat",
        )


def _chat_body(model: str, messages: list[dict], max_tokens: int, response_format: dict | None) -> bytes:
    payload: dict = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return json.dumps(payload).encode("utf-8")


def _is_bedrock(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host.startswith("bedrock-mantle.") or host.startswith("bedrock-runtime.")


def _bedrock_region(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    labels = host.split(".")
    if len(labels) >= 2 and labels[0] in {"bedrock-mantle", "bedrock-runtime"}:
        return labels[1]
    raise RuntimeError("Bedrock region is missing from the chat URL.")


def _signed_bedrock_request(url: str, body: bytes) -> urllib.request.Request:
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials are required to call Gemma on Bedrock.")
    aws_request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials.get_frozen_credentials(), "bedrock", _bedrock_region(url)).add_auth(aws_request)
    return urllib.request.Request(
        url,
        data=body,
        headers={key: value for key, value in aws_request.headers.items()},
        method="POST",
    )
