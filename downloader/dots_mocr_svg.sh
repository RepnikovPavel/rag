#!/bin/bash
CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="rednote-hilab/dots.mocr-svg"
SNAPSHOT_DIR="$CKPTDIR/models--rednote-hilab--dots.mocr-svg/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"

# Создаем папку и проверяем результат
if ! mkdir -p "$SNAPSHOT_DIR" 2>/dev/null; then
    echo "❌ Ошибка: Не удалось создать папку $SNAPSHOT_DIR"
    echo "👉 У вас нет прав на запись в /mnt/nvme/ или диск не примонтирован."
    echo "👉 Запустите скрипт с другим путем: bash $0 ~/models"
    exit 1
fi

# Переходим в папку и проверяем результат
if ! cd "$SNAPSHOT_DIR"; then
    echo "❌ Ошибка: Не удалось перейти в $SNAPSHOT_DIR"
    exit 1
fi

# Основные конфиги, метаданные и кастомный код
echo "📥 Конфигурационные файлы, код модели и метаданные..."
for file in .gitattributes NOTICE README.md chat_template.json config.json \
            configuration_dots.py generation_config.json merges.txt \
            model.safetensors.index.json modeling_dots_ocr.py modeling_dots_vision.py \
            preprocessor_config.json special_tokens_map.json tokenizer.json \
            tokenizer_config.json vocab.json; do
    wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

# Файл лицензии (с пробелами)
echo "📥 Файл лицензии..."
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/dots.mocr%20LICENSE%20AGREEMENT" \
    -O "dots.mocr LICENSE AGREEMENT"

# Веса модели (файлы лежат в Xet-хранилище, но wget корректно заберет их по редиректу)
echo "🚀 Скачиваем 2 safetensors файла (~6.1GB)..."
for i in 1 2; do
    # Форматируем число: 1 превращается в 00001, 2 в 00002
    printf -v NUM "%05d" "$i"
    FILENAME="model-${NUM}-of-00002.safetensors"
    echo "📦 Скачиваем $FILENAME..."
    wget --show-progress --continue \
        "https://huggingface.co/$MODEL_ID/resolve/main/$FILENAME"
done

echo -e "\n✅ Проверка структуры:"
echo "Веса: $(ls *.safetensors 2>/dev/null | wc -l)/2"
echo "Кастомный код: $(ls *.py 2>/dev/null | wc -l) файлов (.py)"
du -sh *.safetensors 2>/dev/null | head -3
echo "🎉 Полная структура для dots.mocr-svg готова!"