import os
import subprocess
import sys


def test_cli_fixture_dry_run() -> None:
    fixture = os.path.join("tests", "fixtures", "issue_opened.json")
    result = subprocess.run(
        [sys.executable, "-m", "kubetag", "--dry-run", "--fixture", fixture],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "Labels:" in result.stdout
    assert "sig/cli" in result.stdout
    assert "area/kubectl" in result.stdout
