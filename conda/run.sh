set -e

MINICONDA_PATH=${1:-~/miniconda3}

if [ ! -f "$MINICONDA_PATH/bin/conda" ]; then
    echo "❌ Miniconda3 не найден: $MINICONDA_PATH"
    echo "💡 ./setup_conda_env.sh [ПУТЬ_К_MINICONDA]"
    exit 1
fi

source $MINICONDA_PATH/bin/activate && conda activate modelscu124