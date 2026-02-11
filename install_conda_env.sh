#!/usr/bin/env bash
set -euo pipefail

# One-click Conda environment setup for Multi-SpinX (macOS/Linux).
# Usage:
#   ./install_conda_env.sh
#   ./install_conda_env.sh <env_name>

ENV_NAME="${1:-multi_spinx}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
  Darwin|Linux)
    ;;
  *)
    echo "Unsupported OS: ${OS_NAME}"
    echo "This installer supports macOS (Darwin) and Linux."
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found in PATH."
  echo "Please install Anaconda or Miniconda first, then re-run this script."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Target environment: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "OS: ${OS_NAME}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' already exists. Updating dependencies..."
  conda install -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    numpy \
    pandas \
    scipy \
    scikit-image \
    matplotlib
else
  echo "Creating environment '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" \
    numpy \
    pandas \
    scipy \
    scikit-image \
    matplotlib
fi

echo
echo "Done."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Installed packages:"
echo "- numpy"
echo "- pandas"
echo "- scipy"
echo "- scikit-image"
echo "- matplotlib"
