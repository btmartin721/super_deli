# Repository layout and local data

The active package is `src/super_deli/`. `tests/` verifies the current APIs;
`examples/` contains synthetic inputs and portable benchmark evidence;
`docs/` describes implemented methods and their limitations; `scripts/` contains
reproducible study/report utilities. The standalone image/TPS manifest utility
remains supported and tested.

Obsolete imputation, training-data download, profiling, generated output and
historical figure trees have been removed from the working repository. Personal
filenames and personal specimen/dataset collections belong outside this tree.
Author/copyright credits and scientific citations remain intact. The original
checkout, including its unresolved index, stashes and local/ignored data, was
archived and verified outside the repository before cleanup. No personal data
or private archive inventory is published with the code.

Use a separate data/output directory and pass its paths explicitly in dataset
configurations. Local cache, virtual-environment, build and data directories are
ignored; published synthetic CSV/PDF fixtures and benchmark evidence remain
version controlled. Scientific results from earlier checkpoints are retained
with their original hashes rather than rewritten to match newer methods.

The default GitHub branch contained only the foundation checkpoint when this
cleanup began. The new draft targets `master` and includes the complete
implementation through the four-group diagnostic checkpoint as well as this
cleanup. Publishing the draft does not update the default branch automatically.
