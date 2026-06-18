from __future__ import annotations

import argparse
import os

from kubetag.application import run_application
from kubetag.config import ConfigurationError, load_config
from kubetag.github.client import GitHubClient, GitHubClientError, parse_issue_reference
from kubetag.logging_config import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Kubernetes GitHub issues")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fixture", "--event-file", dest="fixture")
    parser.add_argument("--title")
    parser.add_argument("--body")
    parser.add_argument("--issue")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if (args.title or args.body) and not args.dry_run:
        parser.error("--title and --body require --dry-run")
    if args.fixture and (args.title or args.body):
        parser.error("--fixture cannot be combined with --title or --body")
    if args.issue and (args.fixture or args.title or args.body):
        parser.error("--issue cannot be combined with other issue inputs")
    if args.issue and not args.dry_run:
        parser.error("--issue requires --dry-run")
    try:
        config = load_config()
    except ConfigurationError as error:
        parser.error(str(error))
    setup_logging(config.log_level)
    title = args.title
    body = args.body
    if args.issue:
        try:
            owner, repo, number = parse_issue_reference(args.issue)
            title, body = GitHubClient(
                os.environ.get("GITHUB_TOKEN"),
                timeout_seconds=config.request_timeout_seconds,
            ).get_issue(owner, repo, number)
        except (ValueError, GitHubClientError) as error:
            parser.error(str(error))
    raise SystemExit(
        run_application(
            title=title,
            body=body,
            event_path=args.fixture,
            dry_run_override=True if args.dry_run else None,
            config=config,
        )
    )


if __name__ == "__main__":
    main()
