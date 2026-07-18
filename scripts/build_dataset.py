from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from kubetag.text_processing import clean_issue_field, prepare_text

API_ROOT = "https://api.github.com"
TARGET_PREFIXES = ("area/", "kind/", "sig/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological KubeTag datasets from closed GitHub issues."
    )
    parser.add_argument("--repository", default="kubernetes/kubernetes")
    parser.add_argument(
        "--schema",
        default="artifacts/model/label_schema.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dataset",
    )
    parser.add_argument("--closed-before", required=True)
    parser.add_argument("--created-after", default="2015-01-01T00:00:00Z")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--lockbox-fraction", type=float, default=0.10)
    parser.add_argument("--max-issues", type=int)
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def load_labels(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("labels")
    if not isinstance(items, list):
        raise ValueError("Label schema requires a labels list")
    labels = tuple(
        item["name"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    )
    if not labels or len(labels) != len(items) or len(labels) != len(set(labels)):
        raise ValueError("Label schema contains invalid or duplicate labels")
    if any(not label.startswith(TARGET_PREFIXES) for label in labels):
        raise ValueError("Label schema contains an unsupported taxonomy")
    return labels


def request_page(
    client: httpx.Client,
    url: str,
    params: dict[str, str | int] | None,
) -> httpx.Response:
    for attempt in range(4):
        try:
            response = client.get(url, params=params)
        except httpx.RequestError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
            continue
        if (
            response.status_code == 403
            and response.headers.get("x-ratelimit-remaining") == "0"
        ):
            reset_at = int(response.headers.get("x-ratelimit-reset", "0"))
            wait_seconds = max(1, reset_at - int(time.time()))
            raise RuntimeError(
                f"GitHub API rate limit reached; retry in {wait_seconds} seconds"
            )
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == 3:
                response.raise_for_status()
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("GitHub request did not complete")


def iter_closed_issues(
    repository: str,
    token: str,
    created_after: datetime,
    closed_before: datetime,
) -> Iterator[dict[str, Any]]:
    if repository.count("/") != 1:
        raise ValueError("Repository must use owner/name format")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "KubeTag dataset builder",
    }
    url: str | None = f"{API_ROOT}/repos/{repository}/issues"
    params: dict[str, str | int] | None = {
        "state": "closed",
        "sort": "created",
        "direction": "asc",
        "per_page": 100,
    }
    with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
        while url:
            response = request_page(client, url, params)
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError("GitHub returned an invalid issues response")
            for issue in payload:
                if not isinstance(issue, dict) or "pull_request" in issue:
                    continue
                created_at = issue.get("created_at")
                closed_at = issue.get("closed_at")
                if not isinstance(created_at, str) or not isinstance(closed_at, str):
                    continue
                created = parse_timestamp(created_at)
                closed = parse_timestamp(closed_at)
                if created > closed_before:
                    return
                if created >= created_after and closed <= closed_before:
                    yield issue
            next_link = response.links.get("next")
            url = next_link.get("url") if next_link else None
            params = None


def normalized_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(r"\b[0-9a-f]{8,}\b", " ", normalized)
    normalized = re.sub(r"\d+", " ", normalized)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def target_column(label: str) -> str:
    return "target_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def issue_labels(issue: dict[str, Any], known_labels: set[str]) -> tuple[str, ...]:
    raw_labels = issue.get("labels")
    if not isinstance(raw_labels, list):
        return ()
    labels = {
        item["name"]
        for item in raw_labels
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"] in known_labels
    }
    return tuple(sorted(labels))


def build_row(
    issue: dict[str, Any],
    repository: str,
    labels: tuple[str, ...],
    known_labels: tuple[str, ...],
) -> dict[str, str | int]:
    number = issue.get("number")
    title = issue.get("title")
    body = issue.get("body")
    created_at = issue.get("created_at")
    html_url = issue.get("html_url")
    if (
        not isinstance(number, int)
        or not isinstance(title, str)
        or body is not None
        and not isinstance(body, str)
        or not isinstance(created_at, str)
    ):
        raise ValueError("GitHub returned an invalid issue")
    clean_title = clean_issue_field(title, known_labels)
    clean_body = clean_issue_field(body, known_labels)
    title_key = normalized_title(clean_title) or f"issue-{number}"
    group_id = "grp_" + hashlib.sha256(title_key.encode("utf-8")).hexdigest()[:16]
    row: dict[str, str | int] = {
        "sample_id": f"github_issue_{number}",
        "issue_number": number,
        "issue_url": (
            html_url
            if isinstance(html_url, str)
            else f"https://github.com/{repository}/issues/{number}"
        ),
        "issue_created_at": created_at,
        "cleaned_issue_title": clean_title,
        "cleaned_issue_body": clean_body,
        "model_input": prepare_text(title, body, known_labels),
        "kind_labels": "|".join(label for label in labels if label.startswith("kind/")),
        "sig_labels": "|".join(label for label in labels if label.startswith("sig/")),
        "area_labels": "|".join(label for label in labels if label.startswith("area/")),
        "retained_labels": "|".join(labels),
        "retained_label_count": len(labels),
        "split_group_id": group_id,
    }
    row.update({target_column(label): int(label in labels) for label in known_labels})
    return row


def quantile(values: list[datetime], fraction: float) -> datetime:
    ordered = sorted(value.timestamp() for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    timestamp = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def assign_splits(
    rows: list[dict[str, str | int]],
    validation_fraction: float,
    lockbox_fraction: float,
) -> None:
    if not rows:
        raise ValueError("No supported labeled issues were collected")
    if validation_fraction <= 0 or lockbox_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if validation_fraction + lockbox_fraction >= 0.5:
        raise ValueError("Validation and lockbox fractions are too large")
    dates = [parse_timestamp(str(row["issue_created_at"])) for row in rows]
    validation_cutoff = quantile(dates, 1.0 - validation_fraction - lockbox_fraction)
    lockbox_cutoff = quantile(dates, 1.0 - lockbox_fraction)
    latest_by_group: dict[str, datetime] = {}
    for row, created_at in zip(rows, dates, strict=True):
        group = str(row["split_group_id"])
        latest_by_group[group] = max(
            created_at,
            latest_by_group.get(group, created_at),
        )
    for row in rows:
        latest = latest_by_group[str(row["split_group_id"])]
        if latest >= lockbox_cutoff:
            row["final_split"] = "lockbox"
        elif latest >= validation_cutoff:
            row["final_split"] = "validation"
        else:
            row["final_split"] = "train"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    output_dir: Path,
    rows: list[dict[str, str | int]],
    labels: tuple[str, ...],
    repository: str,
    created_after: datetime,
    closed_before: datetime,
    validation_fraction: float,
    lockbox_fraction: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    columns = list(rows[0])
    split_details: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "lockbox"):
        path = output_dir / f"{split_name}.csv"
        split_rows = [row for row in rows if row["final_split"] == split_name]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)
        split_details[split_name] = {
            "rows": len(split_rows),
            "groups": len({row["split_group_id"] for row in split_rows}),
            "sha256": file_sha256(path),
        }
    manifest = {
        "repository": repository,
        "created_after": created_after.isoformat(),
        "closed_before": closed_before.isoformat(),
        "split_method": "grouped_creation_time_chronological_strict",
        "validation_fraction": validation_fraction,
        "lockbox_fraction": lockbox_fraction,
        "labels": labels,
        "rows": len(rows),
        "splits": split_details,
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_rows(
    issues: Iterable[dict[str, Any]],
    repository: str,
    labels: tuple[str, ...],
    max_issues: int | None,
) -> list[dict[str, str | int]]:
    known_labels = set(labels)
    rows = []
    for issue in issues:
        retained = issue_labels(issue, known_labels)
        if not retained:
            continue
        rows.append(build_row(issue, repository, retained, labels))
        if max_issues is not None and len(rows) >= max_issues:
            break
    rows.sort(key=lambda row: (str(row["issue_created_at"]), int(row["issue_number"])))
    return rows


def main() -> None:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required")
    labels = load_labels(Path(args.schema))
    created_after = parse_timestamp(args.created_after)
    closed_before = parse_timestamp(args.closed_before)
    issues = iter_closed_issues(
        args.repository,
        token,
        created_after,
        closed_before,
    )
    rows = collect_rows(
        issues,
        args.repository,
        labels,
        args.max_issues,
    )
    assign_splits(rows, args.validation_fraction, args.lockbox_fraction)
    write_dataset(
        Path(args.output_dir),
        rows,
        labels,
        args.repository,
        created_after,
        closed_before,
        args.validation_fraction,
        args.lockbox_fraction,
    )
    print(f"Dataset ready: {args.output_dir} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
