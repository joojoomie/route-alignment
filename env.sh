# Source this before running any command in the reproduction guide.
#   source env.sh
# One interpreter with torch, transformers and scikit-image (see requirements.txt).
# Edit RA_PY for your machine; nothing else in the repo names a path.
export RA_PY="${RA_PY:-python3}"          # torch + transformers + scikit-image
export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/route-alignment-mpl}"
