"""Run the separately frozen fresh unknown-K benchmark CLI."""

import sys

from super_deli.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark-unknown-k-spectral", *sys.argv[1:]]))
