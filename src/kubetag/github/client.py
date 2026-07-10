import logging
import time
from typing import Any, Optional
import httpx

logger = logging.getLogger(__name__)

class GitHubClientError(Exception):
    """Base exception for GitHub API client errors."""
    pass

class GitHubClient:
    """A stateless client to interact with the GitHub Issues API using httpx."""
    
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.github.com",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ) -> None:
        """Initialize the GitHub client.
        
        Args:
            token: The GitHub token for authentication.
            base_url: The GitHub API base URL.
            timeout_seconds: Timeout for requests in seconds.
            max_retries: Maximum number of retries for transient errors.
            backoff_factor: Backoff factor for exponential retry delays.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "KubeTag-bot",
        }

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Any] = None,
    ) -> httpx.Response:
        """Make an HTTP request with retry logic for transient errors and rate limits.
        
        Args:
            method: The HTTP method (GET, POST, PATCH, etc.).
            path: The relative API path (starting with /).
            json_data: Optional JSON payload.
            
        Returns:
            The httpx.Response object.
            
        Raises:
            GitHubClientError: If the request fails after all retries or gets a 4xx error.
        """
        url = f"{self.base_url}{path}"
        
        with httpx.Client(headers=self.headers, timeout=self.timeout) as client:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug("Request %s %s (Attempt %d/%d)", method, url, attempt, self.max_retries)
                    response = client.request(method, url, json=json_data)
                    
                    is_rate_limited_403 = False
                    if response.status_code == 403:
                        rem_header = response.headers.get("x-ratelimit-remaining")
                        body_text = response.text.lower()
                        if rem_header == "0" or "rate limit" in body_text:
                            is_rate_limited_403 = True
                    
                    should_retry = (
                        response.status_code == 429 or 
                        (500 <= response.status_code < 600) or
                        is_rate_limited_403
                    )
                    
                    if should_retry:
                        if attempt == self.max_retries:
                            raise GitHubClientError(
                                f"GitHub request failed after {self.max_retries} attempts. "
                                f"Status {response.status_code}: {response.text}"
                            )
                        
                        sleep_time = self.backoff_factor ** attempt
                        
                        retry_after = response.headers.get("retry-after")
                        if retry_after:
                            try:
                                sleep_time = float(retry_after)
                            except ValueError:
                                pass
                        elif is_rate_limited_403:
                            reset_time = response.headers.get("x-ratelimit-reset")
                            if reset_time:
                                try:
                                    sleep_time = max(0.0, float(reset_time) - time.time())
                                except ValueError:
                                    pass
                                    
                        sleep_time = min(sleep_time, 60.0)
                        
                        logger.warning(
                            "Rate limit or transient error (Status %d) encountered on attempt %d. "
                            "Retrying in %.2fs...",
                            response.status_code, attempt, sleep_time
                        )
                        time.sleep(sleep_time)
                        continue
                    
                    elif response.status_code not in (200, 201):
                        raise GitHubClientError(
                            f"GitHub request failed with non-retryable status {response.status_code}: {response.text}"
                        )
                    
                    return response
                    
                except httpx.RequestError as e:
                    if attempt == self.max_retries:
                        logger.error("Request failed after max retries: %s", e)
                        raise GitHubClientError(f"GitHub request failed after {self.max_retries} attempts: {e}") from e
                    sleep_time = min(self.backoff_factor ** attempt, 60.0)
                    logger.warning("Request error encountered: %s. Retrying in %.2fs...", e, sleep_time)
                    time.sleep(sleep_time)
                    
            raise GitHubClientError("Failed to perform request due to retry logic exhaustion.")

    def get_repo_labels(self, owner: str, repo: str) -> list[str]:
        """Fetch all existing labels in a repository.
        
        GET /repos/{owner}/{repo}/labels
        """
        path = f"/repos/{owner}/{repo}/labels?per_page=100"
        response = self._request("GET", path)
        
        if response.status_code != 200:
            raise GitHubClientError(
                f"Failed to fetch repository labels (Status {response.status_code}): {response.text}"
            )
            
        try:
            result = response.json()
            if isinstance(result, list):
                return [item.get("name", "") for item in result if isinstance(item, dict)]
            return []
        except Exception as e:
            raise GitHubClientError(f"Failed to parse repository labels: {e}") from e

    def add_labels_to_issue(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> list[str]:
        """Add labels to a specific issue.
        
        POST /repos/{owner}/{repo}/issues/{issue_number}/labels
        """
        path = f"/repos/{owner}/{repo}/issues/{issue_number}/labels"
        response = self._request("POST", path, json_data={"labels": labels})
        
        if response.status_code != 200:
            raise GitHubClientError(
                f"Failed to add labels (Status {response.status_code}): {response.text}"
            )
            
        try:
            result = response.json()
            if isinstance(result, list):
                return [item.get("name", "") for item in result if isinstance(item, dict)]
            return labels
        except Exception as e:
            raise GitHubClientError(f"Failed to parse label application response: {e}") from e
