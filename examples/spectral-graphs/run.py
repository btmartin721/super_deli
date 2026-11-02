"""Run the versioned diagnostic CLI; supplied K is not species discovery."""

import sys

from super_deli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["diagnose-spectral-graphs", *sys.argv[1:]]))
