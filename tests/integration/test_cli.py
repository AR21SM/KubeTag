import os
import subprocess
import sys

def test_cli_integration_dry_run() -> None:
    # Get the path to the fixture file
    fixture_path = os.path.join("tests", "fixtures", "issue_opened.json")
    assert os.path.exists(fixture_path), f"Fixture not found at {fixture_path}"
    
    # Run the package as a module in dry-run mode with the fixture
    cmd = [
        sys.executable,
        "-m",
        "kubetag",
        "--dry-run",
        "--fixture",
        fixture_path
    ]
    
    # Run process and capture stdout/stderr
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    
    # Print outputs for debugging in case of test failure
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    
    # Assert exit code is 0
    assert result.returncode == 0
    
    # Verify the generated output in stdout
    assert "Labels: ['sig/cli', 'sig/auth', 'sig/scheduling', 'area/kubectl']" in result.stdout
    assert "Triage result: Applying labels:" in result.stdout
    assert "Model Version:" in result.stdout
