"""Forward one approved study lifecycle stage to the installed CLI."""

import sys

from super_deli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark-matched-neural", *sys.argv[1:]]))
