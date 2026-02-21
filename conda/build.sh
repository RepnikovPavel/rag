#!/bin/bash
# setup_conda_env.sh - Установка через conda + pip для PyTorch

set -e

# Путь к Miniconda (аргумент с дефолтом)
MINICONDA_PATH=${1:-~/miniconda3}

echo "🚀 Устанавливаем окружение: $MINICONDA_PATH"

# Проверяем Miniconda
if [ ! -f "$MINICONDA_PATH/bin/conda" ]; then
    echo "❌ Miniconda3 не найден: $MINICONDA_PATH"
    echo "💡 ./setup_conda_env.sh [ПУТЬ_К_MINICONDA]"
    exit 1
fi

# Инициализируем conda
source "$MINICONDA_PATH/etc/profile.d/conda.sh"
source "$MINICONDA_PATH/bin/activate"

conda --version || { echo "❌ Ошибка conda"; exit 1; }

# Обновляем conda
echo "📦 Обновляем conda..."
conda update -n base conda -y

# Создаем окружение
ENV_NAME="modelscu124"
echo "📦 Создаем: $ENV_NAME (Python 3.10)..."
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# ❌ НЕ ставим PyTorch через conda (проблемы с версиями)
echo "⚠️  Пропускаем conda PyTorch (torchvision==0.21.0 недоступен)"

# ✅ PyTorch через pip (как в Dockerfile)
echo "🔥 PyTorch 2.6.0 + CUDA 12.4 через pip..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# # Системные пакеты (conda)
# echo "📚 Системные зависимости..."
# conda install -c conda-forge \
#     cmake git curl ninja tk htop graphviz gnuplot tree \
#     build-essential libgcc openssl boost-cpp bzip2 \
#     freeglut mesa-libgl-devel libx11 libxrender libxi \
#     libxcb libxkbcommon \
#     -y

# Python пакеты
echo "🐍 Остальные Python пакеты..."
pip install simple-colors wheel

# CUDA переменные
echo "⚙️ CUDA окружение..."
ENV_DIR=$(conda info --base)/envs/$ENV_NAME
mkdir -p $ENV_DIR/etc/conda/activate.d $ENV_DIR/etc/conda/deactivate.d

cat > $ENV_DIR/etc/conda/activate.d/cuda_env.sh << 'EOF'
#!/bin/bash
export CUDAV=12.4
export CUDA_HOME=/usr/local/cuda-$CUDAV
export CUDA_ROOT=/usr/local/cuda-$CUDAV
export CUDA_INCLUDE_DIRS=/usr/local/cuda-$CUDAV/targets/x86_64-linux/include/
export CUDA_CUDART_LIBRARY=/usr/local/cuda-$CUDAV/targets/x86_64-linux/lib/
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-$CUDAV/
export CMAKE_CUDA_COMPILER=/usr/local/cuda-$CUDAV/bin/nvcc
export CUDACXX=/usr/local/cuda-$CUDAV/bin/nvcc
export PATH=/usr/local/cuda-$CUDAV/bin:$PATH
export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;9.0+PTX"
EOF

cat > $ENV_DIR/etc/conda/deactivate.d/cuda_env.sh << 'EOF'
#!/bin/bash
export PATH=$(echo $PATH | sed -E 's;:/usr/local/cuda-[^:]+;;g')
EOF

chmod +x $ENV_DIR/etc/conda/activate.d/cuda_env.sh
chmod +x $ENV_DIR/etc/conda/deactivate.d/cuda_env.sh

# Тест
echo "🧪 Тестируем PyTorch..."
python -c "
import torch
import torchvision
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Torchvision: {torchvision.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()} ({torch.version.cuda})')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "🎉 Готово!"
echo "source $MINICONDA_PATH/bin/activate && conda activate modelscu124"
echo "TensorRT: ручная установка с NVIDIA сайта"
