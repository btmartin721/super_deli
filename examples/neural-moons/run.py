"""Use the installed versioned diagnostic CLI; see docs/neural-moons.md."""

import sys

from super_deli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose-neural-moons", *sys.argv[1:]]))
