from unittest.mock import MagicMock, patch

import httpx
import pytest

from kubetag.github.client import GitHubClient, GitHubClientError, parse_issue_reference


def _response(status=200, payload=None, text="", headers=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status
    response.json.return_value = payload
    response.text = text
    response.headers = headers or {}
    return response


def test_parses_issue_reference() -> None:
    assert parse_issue_reference("kubernetes/kubernetes#123") == (
        "kubernetes",
        "kubernetes",
        123,
    )
    with pytest.raises(ValueError, match="owner/repository#number"):
        parse_issue_reference("invalid")


@patch("httpx.Client")
def test_fetches_public_issue(client_class) -> None:
    client_class.return_value.__enter__.return_value.request.return_value = _response(
        payload={"title": "Node failure", "body": "Kubelet stopped"}
    )
    client = GitHubClient()
    assert client.get_issue("kubernetes", "kubernetes", 42) == (
        "Node failure",
        "Kubelet stopped",
    )


@patch("httpx.Client")
def test_paginates_repository_labels(client_class) -> None:
    first = [{"name": f"label-{index}"} for index in range(100)]
    second = [{"name": "kind/bug"}]
    request = client_class.return_value.__enter__.return_value.request
    request.side_effect = [_response(payload=first), _response(payload=second)]
    labels = GitHubClient("token").get_repo_labels("owner", "repo")
    assert len(labels) == 101
    assert labels[-1] == "kind/bug"
    assert request.call_count == 2


@patch("httpx.Client")
def test_adds_labels(client_class) -> None:
    request = client_class.return_value.__enter__.return_value.request
    request.return_value = _response(payload=[{"name": "kind/bug"}])
    labels = GitHubClient("token").add_labels_to_issue(
        "owner", "repo", 42, ["kind/bug"]
    )
    assert labels == ["kind/bug"]
    request.assert_called_once_with(
        "POST",
        "https://api.github.com/repos/owner/repo/issues/42/labels",
        json={"labels": ["kind/bug"]},
    )


@patch("time.sleep")
@patch("httpx.Client")
def test_retries_server_error(client_class, sleep) -> None:
    request = client_class.return_value.__enter__.return_value.request
    request.side_effect = [
        _response(status=500),
        _response(payload=[{"name": "kind/bug"}]),
    ]
    labels = GitHubClient("token").add_labels_to_issue(
        "owner", "repo", 42, ["kind/bug"]
    )
    assert labels == ["kind/bug"]
    sleep.assert_called_once()


@patch("httpx.Client")
def test_rejects_non_retryable_error(client_class) -> None:
    client_class.return_value.__enter__.return_value.request.return_value = _response(
        status=403,
        text="Write permission required",
    )
    with pytest.raises(GitHubClientError, match="non-retryable status 403"):
        GitHubClient("token").add_labels_to_issue("owner", "repo", 42, ["kind/bug"])
