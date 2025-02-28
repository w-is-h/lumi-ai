#!/usr/bin/env python
"""Main entry point for Lumi package."""

try:
    # When installed as package
    from lumi.cli.s2t_cli import main
except ImportError:
    # When run from source
    from cli.s2t_cli import main

if __name__ == "__main__":
    main()
