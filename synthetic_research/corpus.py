import hashlib
import json
import re

from synthetic_research.gates import PHASES, ResearchError
from synthetic_research.retrieve import SOURCE_KINDS

EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}")


def campaign_sources(campaign: dict) -> list[dict]:
    sources = [dict(document) for document in campaign.get("documents") or []]
    sources.extend(materialize_datasets(campaign))
    return sources


def materialize_datasets(campaign: dict) -> list[dict]:
    rows = []
    for dataset in campaign.get("datasets") or []:
        dataset_id = dataset.get("id") or ""
        for row in dataset.get("rows") or []:
            area = row.get("area") or ""
            rows.append(
                {
                    "id": row.get("id"),
                    "kind": "sales",
                    "cohortIds": list(row.get("cohortIds") or []),
                    "title": row.get("title") or f"Filter coffee sales, {area}",
                    "text": row.get("text") or "",
                    "datasetId": dataset_id,
                    "area": area,
                    "channel": row.get("channel") or "",
                    "cups": row.get("cups"),
                    "avgTicketInr": row.get("avgTicketInr"),
                }
            )
    return rows


def corpus_hash(campaign: dict) -> str:
    rows = [
        {
            "id": document["id"],
            "kind": document["kind"],
            "cohortIds": sorted(document.get("cohortIds") or []),
            "text": document.get("text") or "",
        }
        for document in campaign_sources(campaign)
    ]
    rows.sort(key=lambda row: row["id"])
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_pii(documents: list[dict]) -> list[str]:
    findings = []
    for document in documents:
        text = f"{document.get('title', '')}\n{document.get('text', '')}"
        if EMAIL.search(text) or PHONE.search(text):
            findings.append(document.get("id") or "unknown")
    return findings


def validate_campaign(campaign: dict) -> None:
    errors = []
    if not campaign.get("campaignId"):
        errors.append("campaignId is required.")
    if not campaign.get("grantId"):
        errors.append("grantId is required.")
    kit = campaign.get("kit") or {}
    if not kit.get("locked"):
        errors.append("kit.locked must be true.")
    for field in ("version", "brandAlias", "purpose"):
        if not kit.get(field):
            errors.append(f"kit.{field} is required.")
    if not (campaign.get("marketPack") or {}).get("market"):
        errors.append("marketPack.market is required.")
    cohorts = campaign.get("cohorts") or []
    if not cohorts:
        errors.append("At least one cohort rule is required.")
    cohort_ids = set()
    for cohort in cohorts:
        cohort_id = cohort.get("id")
        if not cohort_id or cohort_id in cohort_ids:
            errors.append("Each cohort needs a unique id.")
        cohort_ids.add(cohort_id)
        for field in ("cardId", "label", "rule", "tension"):
            if not cohort.get(field):
                errors.append(f"Cohort {cohort_id} is missing {field}.")
        if not cohort.get("bounds"):
            errors.append(f"Cohort {cohort_id} needs screener bounds.")
    phases = set(PHASES)
    mains = (campaign.get("guide") or {}).get("mains") or []
    if not mains:
        errors.append("The guide needs locked main questions.")
    for main in mains:
        if main.get("phase") not in phases or not main.get("text"):
            errors.append("Each main question needs a known phase and text.")
    documents = campaign.get("documents") or []
    seen = set()
    for document in documents:
        document_id = document.get("id")
        if not document_id or document_id in seen:
            errors.append("Each gold document needs a unique id.")
        seen.add(document_id)
        if document.get("kind") not in SOURCE_KINDS - {"sales"}:
            errors.append(f"{document_id} has a kind outside transcript, review, and screener.")
        if not document.get("text"):
            errors.append(f"{document_id} has no text.")
        if not document.get("cohortIds"):
            errors.append(f"{document_id} is not tied to a cohort.")
    for cohort_id in cohort_ids:
        covered = any(
            document.get("kind") == "screener" and cohort_id in (document.get("cohortIds") or []) and document.get("members")
            for document in documents
        )
        if not covered:
            errors.append(f"Cohort {cohort_id} has no screener membership count.")
    _validate_datasets(campaign, cohort_ids, seen, errors)
    if errors:
        raise ResearchError("PACK_INVALID", " ".join(errors))


def _validate_datasets(campaign: dict, cohort_ids: set, seen: set, errors: list[str]) -> None:
    dataset_ids = set()
    for dataset in campaign.get("datasets") or []:
        dataset_id = dataset.get("id")
        if not dataset_id or dataset_id in dataset_ids:
            errors.append("Each additional dataset needs a unique id.")
        dataset_ids.add(dataset_id)
        if not dataset.get("label"):
            errors.append(f"Dataset {dataset_id} needs a label.")
        rows = dataset.get("rows") or []
        if not rows:
            errors.append(f"Dataset {dataset_id} has no rows.")
        for row in rows:
            row_id = row.get("id")
            if not row_id or row_id in seen:
                errors.append("Each sales row needs a unique id.")
            seen.add(row_id)
            if not row.get("area"):
                errors.append(f"{row_id} has no office area.")
            if not row.get("text"):
                errors.append(f"{row_id} has no text.")
            if row.get("cups") is None or row.get("avgTicketInr") is None:
                errors.append(f"{row_id} needs cups and avgTicketInr.")
            tied = row.get("cohortIds") or []
            if not tied:
                errors.append(f"{row_id} is not tied to a cohort.")
            unknown = [cohort_id for cohort_id in tied if cohort_id not in cohort_ids]
            if unknown:
                errors.append(f"{row_id} cites an unknown cohort.")
