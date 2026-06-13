import argparse
import sys
from kubetag.application import run_application
from kubetag.config import load_config
from kubetag.logging_config import setup_logging

def main() -> None:
    """CLI entrypoint for KubeTag."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    config = load_config()
    setup_logging(config.log_level)
    
    parser = argparse.ArgumentParser(
        description="KubeTag CLI for GHA and local dry-runs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: logs labels to apply and skips GitHub API calls."
    )
    parser.add_argument(
        "--fixture",
        "--event-file",
        dest="fixture",
        type=str,
        help="Path to a GitHub issues event payload JSON file."
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Raw issue title to use for dry-run (requires --dry-run)."
    )
    parser.add_argument(
        "--body",
        type=str,
        help="Raw issue body to use for dry-run (requires --dry-run)."
    )
    
    args = parser.parse_args()
    
    if (args.title or args.body) and not args.dry_run:
        parser.error("--title and --body can only be used with --dry-run.")
        
    if args.fixture and (args.title or args.body):
        parser.error("Cannot combine --fixture with --title or --body.")

    exit_code = run_application(
        title=args.title,
        body=args.body,
        event_path=args.fixture,
        dry_run_override=True if args.dry_run else None,
    )
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
