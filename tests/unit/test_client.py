from unittest.mock import MagicMock, patch
import httpx
import pytest
from kubetag.github.client import GitHubClient, GitHubClientError

@patch("httpx.Client")
def test_add_labels_to_issue_success(mock_client_class) -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = [{"name": "kind/bug"}, {"name": "sig/auth"}]
    
    mock_client = MagicMock()
    mock_client.request.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    client = GitHubClient(token="fake-token")
    result = client.add_labels_to_issue("owner", "repo", 42, ["kind/bug", "sig/auth"])
    
    assert result == ["kind/bug", "sig/auth"]
    mock_client.request.assert_called_once_with(
        "POST",
        "https://api.github.com/repos/owner/repo/issues/42/labels",
        json={"labels": ["kind/bug", "sig/auth"]}
    )

@patch("httpx.Client")
def test_client_fails_on_400_no_retries(mock_client_class) -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    
    mock_client = MagicMock()
    mock_client.request.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    client = GitHubClient(token="fake-token", max_retries=3, backoff_factor=0.01)
    
    with pytest.raises(GitHubClientError, match="non-retryable status 400"):
        client.add_labels_to_issue("owner", "repo", 42, ["kind/bug"])
        
    assert mock_client.request.call_count == 1

@patch("httpx.Client")
@patch("time.sleep")
def test_client_retries_on_500_and_succeeds(mock_sleep, mock_client_class) -> None:
    mock_resp_fail = MagicMock(spec=httpx.Response)
    mock_resp_fail.status_code = 500
    mock_resp_fail.headers = {}
    mock_resp_fail.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Error", request=MagicMock(), response=mock_resp_fail
    )
    
    mock_resp_ok = MagicMock(spec=httpx.Response)
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = [{"name": "kind/bug"}]
    
    mock_client = MagicMock()
    mock_client.request.side_effect = [mock_resp_fail, mock_resp_ok]
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    client = GitHubClient(token="fake-token", max_retries=3, backoff_factor=0.01)
    result = client.add_labels_to_issue("owner", "repo", 42, ["kind/bug"])
    
    assert result == ["kind/bug"]
    assert mock_client.request.call_count == 2
    mock_sleep.assert_called_once()

@patch("httpx.Client")
@patch("time.sleep")
def test_client_retries_on_rate_limited_403(mock_sleep, mock_client_class) -> None:
    # First response: 403 rate limit
    mock_resp_fail = MagicMock(spec=httpx.Response)
    mock_resp_fail.status_code = 403
    mock_resp_fail.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}
    mock_resp_fail.text = "api rate limit exceeded"
    
    # Second response: success
    mock_resp_ok = MagicMock(spec=httpx.Response)
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = [{"name": "kind/bug"}]
    
    mock_client = MagicMock()
    mock_client.request.side_effect = [mock_resp_fail, mock_resp_ok]
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    client = GitHubClient(token="fake-token", max_retries=3, backoff_factor=0.01)
    result = client.add_labels_to_issue("owner", "repo", 42, ["kind/bug"])
    
    assert result == ["kind/bug"]
    assert mock_client.request.call_count == 2
    mock_sleep.assert_called_once()

@patch("httpx.Client")
def test_client_fails_on_normal_403_no_retries(mock_client_class) -> None:
    # Normal 403: Forbidden (insufficient permission), should fail fast
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 403
    mock_response.headers = {}
    mock_response.text = "Write permission required"
    
    mock_client = MagicMock()
    mock_client.request.return_value = mock_response
    mock_client_class.return_value.__enter__.return_value = mock_client
    
    client = GitHubClient(token="fake-token", max_retries=3, backoff_factor=0.01)
    
    with pytest.raises(GitHubClientError, match="non-retryable status 403"):
        client.add_labels_to_issue("owner", "repo", 42, ["kind/bug"])
        
    assert mock_client.request.call_count == 1

