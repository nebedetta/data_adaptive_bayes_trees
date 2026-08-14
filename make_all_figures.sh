#!/bin/bash
# Regenerate every figure in the paper into results/figures/.
#
#   ./make_all_figures.sh
#
# The six simulation figures run from committed data in about a minute and need
# nothing external. The flow-cytometry figure needs data that is not distributed
# with this repository; it is skipped with an explanation if that data is absent,
# and the published version stays in place. See data/flow/README.md.
#
# Requires the Python packages in requirements.txt. Set PYTHON to choose an
# interpreter:  PYTHON=python3.13 ./make_all_figures.sh

set -u

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Regenerating figures with $($PYTHON --version 2>&1)"
echo

failed=0

run() {
    local description="$1"; shift
    echo "--- $description"
    if "$PYTHON" "$@"; then
        echo
    else
        echo "    FAILED: $*"
        echo
        failed=1
    fi
}

run "1D metric grids (L1, L2)"        src/make_figures/make_1D_figures.py
run "2D metric grids (L1, L2)"        src/make_figures/make_2D_metric_figures.py
run "2D posterior means, iteration 15" src/make_figures/make_2D_posterior_figure.py
run "3D metric grid (L1 and L2)"      src/make_figures/make_3D_figure.py

# Needs a cached fit from src/data_analysis/fit_flow.py, which in turn needs the
# undistributed flow-cytometry data. plot_flow.py explains and exits cleanly
# when the cache is missing, so a clean checkout still completes.
echo "--- flow-cytometry application"
if "$PYTHON" src/data_analysis/plot_flow.py; then
    echo
else
    echo "    (skipped -- see data/flow/README.md; the published figure is"
    echo "     committed at results/figures/2D_CD45RO_CD27_betatree.jpg)"
    echo
fi

if [ "$failed" -ne 0 ]; then
    echo "Some figures failed to build."
    exit 1
fi

echo "Figures written to results/figures/:"
ls -1 results/figures/
