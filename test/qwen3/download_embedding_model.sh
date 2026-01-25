#!/bin/bash
CKPTDIR=${1:-"./models"}
MODEL_ID="Qwen/Qwen3-Embedding-8B"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-Embedding-8B/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

echo "📥 Конфигурационные файлы..."
for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors.index.json; do
    wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

echo "🚀 Скачиваем 4 safetensors файла (~16GB)..."
for i in 01 02 03 04; do
    echo "📦 Скачиваем model-000${i}-of-00004.safetensors"
    wget --show-progress --continue \
        "https://huggingface.co/$MODEL_ID/resolve/main/model-000${i}-of-00004.safetensors"
done

echo -e "\n✅ Проверка:"
echo "Файлов весов: $(ls *.safetensors 2>/dev/null | wc -l)"
du -sh *.safetensors 2>/dev/null | head -5
echo "🎉 Готово для Python: python script.py --ckptdir $CKPTDIR"
