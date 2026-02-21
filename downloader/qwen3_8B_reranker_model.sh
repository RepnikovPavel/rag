#!/bin/bash
CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-Reranker-8B"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-Reranker-8B/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

# Основные конфиги (reranker НЕ использует SentenceTransformers структуру)
echo "📥 Конфигурационные файлы..."
for file in config.json generation_config.json tokenizer.json \
            tokenizer_config.json model.safetensors.index.json merges.txt; do
    wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

# Веса модели - 5 файлов (~16.5GB)
echo "🚀 Скачиваем 5 safetensors файлов (~16.5GB)..."
for i in 01 02 03 04 05; do
    echo "📦 Скачиваем model-000${i}-of-00005.safetensors"
    wget --show-progress --continue \
        "https://huggingface.co/$MODEL_ID/resolve/main/model-000${i}-of-00005.safetensors"
done

# Дополнительные файлы (опционально)
echo "📥 Дополнительные файлы..."
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/vocab.json"

echo -e "\n✅ Проверка структуры:"
echo "Веса: $(ls *.safetensors 2>/dev/null | wc -l)/5"
echo "Конфиги: $(ls *.json 2>/dev/null | wc -l)"
du -sh *.safetensors 2>/dev/null | head -6
echo "🎉 Готово для CrossEncoder reranker!"
