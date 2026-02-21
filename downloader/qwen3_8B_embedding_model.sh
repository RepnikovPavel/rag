#!/bin/bash
CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-Embedding-8B"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-Embedding-8B/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

# Основные конфиги
echo "📥 Конфигурационные файлы (корень)..."
for file in config.json config_sentence_transformers.json tokenizer.json \
            tokenizer_config.json special_tokens_map.json model.safetensors.index.json \
            modules.json merges.txt generation_config.json; do
    wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

# Папка 1_Pooling - КРИТИЧНО!
echo "📥 Pooling конфигурация (1_Pooling/)..."
mkdir -p 1_Pooling
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/1_Pooling/config.json" \
    -O 1_Pooling/config.json

# Веса модели
echo "🚀 Скачиваем 4 safetensors файла (~16GB)..."
for i in 01 02 03 04; do
    echo "📦 model-000${i}-of-00004.safetensors"
    wget --show-progress --continue \
        "https://huggingface.co/$MODEL_ID/resolve/main/model-000${i}-of-00004.safetensors"
done

echo -e "\n✅ Проверка структуры:"
echo "Веса: $(ls *.safetensors 2>/dev/null | wc -l)/4"
echo "Конфиги: $(ls *config*.json modules.json 2>/dev/null | wc -l)"
echo "Pooling: $([[ -f 1_Pooling/config.json ]] && echo '✅' || echo '❌')"

du -sh *.safetensors 1_Pooling/ 2>/dev/null | head -6
echo "🎉 Полная структура для SentenceTransformers готова!"
