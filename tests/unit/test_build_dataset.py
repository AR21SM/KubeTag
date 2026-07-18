from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.build_dataset import (
    assign_splits,
    build_row,
    collect_rows,
    issue_labels,
)

LABELS = ("kind/bug", "kind/failing-test", "sig/node")


def issue(
    number: int,
    title: str,
    created_at: str,
    labels: list[str],
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "body": "/sig node\n`/kind bug`\nThe kubelet test fails.",
        "created_at": created_at,
        "closed_at": created_at,
        "html_url": f"https://github.com/kubernetes/kubernetes/issues/{number}",
        "labels": [{"name": label} for label in labels],
    }


def test_build_row_removes_label_leakage() -> None:
    payload = issue(
        1,
        "[sig-node] kubelet test fails",
        "2024-01-01T00:00:00Z",
        ["kind/failing-test", "sig/node", "priority/important-soon"],
    )
    retained = issue_labels(payload, set(LABELS))
    row = build_row(payload, "kubernetes/kubernetes", retained, LABELS)

    assert retained == ("kind/failing-test", "sig/node")
    assert "sig/node" not in str(row["model_input"]).lower()
    assert "/kind bug" not in str(row["model_input"]).lower()
    assert row["target_kind_failing_test"] == 1
    assert row["target_kind_bug"] == 0


def test_collect_rows_excludes_zero_label_issues() -> None:
    issues = [
        issue(1, "Unlabeled issue", "2024-01-01T00:00:00Z", []),
        issue(2, "Node failure", "2024-01-02T00:00:00Z", ["kind/bug"]),
    ]

    rows = collect_rows(issues, "kubernetes/kubernetes", LABELS, None)

    assert [row["issue_number"] for row in rows] == [2]


def test_assign_splits_keeps_repeated_titles_together() -> None:
    titles = [
        "Alpha failure",
        "Bravo failure",
        "Charlie failure",
        "Repeated failure",
        "Delta failure",
        "Echo failure",
        "Foxtrot failure",
        "Golf failure",
        "Hotel failure",
        "Repeated failure",
    ]
    rows = [
        build_row(
            issue(
                number,
                titles[number - 1],
                f"2024-01-{number:02d}T00:00:00Z",
                ["kind/bug"],
            ),
            "kubernetes/kubernetes",
            ("kind/bug",),
            LABELS,
        )
        for number in range(1, 11)
    ]

    assign_splits(rows, 0.2, 0.1)

    repeated = [row["final_split"] for row in rows if row["issue_number"] in {4, 10}]
    assert repeated == ["lockbox", "lockbox"]
    assert {row["final_split"] for row in rows} == {"train", "validation", "lockbox"}


def test_closed_before_contract_is_timezone_aware() -> None:
    cutoff = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    assert cutoff.tzinfo == timezone.utc
