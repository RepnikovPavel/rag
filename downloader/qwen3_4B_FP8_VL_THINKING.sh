#!/bin/bash
CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-VL-4B-Thinking-FP8"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-VL-4B-Thinking-FP8/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

# Основные конфиги, токенизатор и VL-специфичные файлы
echo "📥 Конфигурационные файлы и токенизатор (корень)..."
for file in .gitattributes README.md chat_template.json config.json generation_config.json \
            model.safetensors.index.json preprocessor_config.json tokenizer.json \
            tokenizer_config.json video_preprocessor_config.json vocab.json; do
    wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

# Веса модели
echo "🚀 Скачиваем 2 safetensors файла (~6GB)..."
for i in 01 02; do
    echo "📦 model-000${i}-of-00002.safetensors"
    wget --show-progress --continue \
        "https://huggingface.co/$MODEL_ID/resolve/main/model-000${i}-of-00002.safetensors"
done

echo -e "\n✅ Проверка структуры:"
echo "Веса: $(ls *.safetensors 2>/dev/null | wc -l)/2"
echo "Конфиги: $(ls config.json generation_config.json model.safetensors.index.json preprocessor_config.json video_preprocessor_config.json 2>/dev/null | wc -l)"
echo "Токенизатор: $([[ -f tokenizer.json ]] && echo '✅' || echo '❌')"

du -sh *.safetensors 2>/dev/null | head -3
echo "🎉 Полная структура модели Qwen3-VL-4B-Thinking-FP8 готова!"