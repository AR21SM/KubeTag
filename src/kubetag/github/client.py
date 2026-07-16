from __future__ import annotations

import re
import time
from typing import Any

import httpx


class GitHubClientError(Exception):
    pass


def parse_issue_reference(reference: str) -> tuple[str, str, int]:
    match = re.fullmatch(r"([^/\s]+)/([^#\s]+)#(\d+)", reference.strip())
    if not match:
        raise ValueError("Issue must use owner/repository#number")
    return match.group(1), match.group(2), int(match.group(3))


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "KubeTag",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _request(
        self,
        method: str,
        path: str,
        json_data: Any | None = None,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        with httpx.Client(headers=self.headers, timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = client.request(method, url, json=json_data)
                except httpx.RequestError as error:
                    if attempt == self.max_retries:
                        raise GitHubClientError(
                            f"GitHub request failed after {self.max_retries} attempts: {error}"
                        ) from error
                    time.sleep(min(self.backoff_factor**attempt, 60.0))
                    continue

                rate_limited = response.status_code == 403 and (
                    response.headers.get("x-ratelimit-remaining") == "0"
                    or "rate limit" in response.text.lower()
                )
                retryable = (
                    response.status_code == 429
                    or rate_limited
                    or response.status_code >= 500
                )
                if retryable and attempt < self.max_retries:
                    delay = response.headers.get("retry-after")
                    sleep_seconds = (
                        float(delay)
                        if delay and delay.replace(".", "", 1).isdigit()
                        else self.backoff_factor**attempt
                    )
                    time.sleep(min(sleep_seconds, 60.0))
                    continue
                if retryable:
                    raise GitHubClientError(
                        f"GitHub request failed after {self.max_retries} attempts: "
                        f"status {response.status_code}"
                    )
                if response.status_code not in {200, 201}:
                    raise GitHubClientError(
                        f"GitHub request failed with non-retryable status "
                        f"{response.status_code}: {response.text}"
                    )
                return response
        raise GitHubClientError("GitHub request did not complete")

    def get_issue(
        self, owner: str, repo: str, issue_number: int
    ) -> tuple[str, str | None]:
        payload = self._request(
            "GET", f"/repos/{owner}/{repo}/issues/{issue_number}"
        ).json()
        if not isinstance(payload, dict) or "pull_request" in payload:
            raise GitHubClientError("The requested resource is not an issue")
        title = payload.get("title")
        body = payload.get("body")
        if not isinstance(title, str) or body is not None and not isinstance(body, str):
            raise GitHubClientError("GitHub returned an invalid issue payload")
        return title, body

    def get_repo_labels(self, owner: str, repo: str) -> list[str]:
        labels: list[str] = []
        page = 1
        while True:
            payload = self._request(
                "GET",
                f"/repos/{owner}/{repo}/labels?per_page=100&page={page}",
            ).json()
            if not isinstance(payload, list):
                raise GitHubClientError("GitHub returned an invalid label payload")
            labels.extend(
                item["name"]
                for item in payload
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
            if len(payload) < 100:
                return labels
            page += 1

    def add_labels_to_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        labels: list[str],
    ) -> list[str]:
        payload = self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/labels",
            json_data={"labels": labels},
        ).json()
        if not isinstance(payload, list):
            raise GitHubClientError("GitHub returned an invalid label response")
        return [
            item["name"]
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
