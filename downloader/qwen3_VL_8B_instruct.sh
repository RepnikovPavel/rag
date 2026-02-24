#!/bin/bash

# Аргумент 1: Путь к папке с моделями (дефолтное значение)
DEST_DIR="${1:-/mnt/nvme/huggingface}"

# Название модели
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"

# Формируем пути
# Имя папки в стиле cache: models--Организация--Модель
CACHE_FOLDER_NAME="models--Qwen--Qwen3-VL-8B-Instruct"
# Путь внутри snapshots (обычно используется ветка main)
TARGET_DIR="$DEST_DIR/$CACHE_FOLDER_NAME/snapshots/main"

# Создаем структуру папок
echo "Создание директории: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Базовый URL для скачивания "сырых" файлов
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

# Список файлов для скачивания (составлен на основе вашего запроса)
FILES=(
    ".gitattributes"
    "README.md"
    "chat_template.json"
    "config.json"
    "generation_config.json"
    "merges.txt"
    "model-00001-of-00004.safetensors"
    "model-00002-of-00004.safetensors"
    "model-00003-of-00004.safetensors"
    "model-00004-of-00004.safetensors"
    "model.safetensors.index.json"
    "preprocessor_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "video_preprocessor_config.json"
    "vocab.json"
)

echo "Начинаем скачивание модели $MODEL_ID в $TARGET_DIR..."

# Цикл скачивания
for FILE in "${FILES[@]}"; do
    echo " -> Скачивание: $FILE"
    # Используем wget:
    # -c : докачка файла, если прервалось
    # -q : тихий режим (можно убрать для подробного вывода)
    # --show-progress : показывать прогресс даже в тихом режиме
    wget -c --show-progress "${BASE_URL}/${FILE}" -O "${TARGET_DIR}/${FILE}"
    
    # Проверка на ошибки
    if [ $? -ne 0 ]; then
        echo "Ошибка при скачивании файла: $FILE"
    fi
done

echo "=========================================="
echo "Скачивание завершено!"
echo "Файлы находятся в: $TARGET_DIR"