# Source this before running any command in the reproduction guide.
#   source env.sh
# Two interpreters are needed and the split is real (see requirements.txt).
# Edit these two lines for your machine; nothing else in the repo names a path.
export RA_PY="${RA_PY:-python3}"          # torch + transformers + scikit-image
export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/route-alignment-mpl}"
