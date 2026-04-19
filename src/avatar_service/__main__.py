"""CLI entry point: ``avatar-service <command>``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avatar-service")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Run the HTTP control plane + avatar pipeline (uvicorn).")
    sub.add_parser("version", help="Print the package version and exit.")

    ns = parser.parse_args(argv)
    if ns.command == "serve":
        from .main import run_uvicorn

        run_uvicorn()
        return 0
    if ns.command == "version":
        from . import __version__

        print(__version__)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
