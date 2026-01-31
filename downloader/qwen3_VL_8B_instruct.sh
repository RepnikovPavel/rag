#!/bin/bash
CKPTDIR=${1:-"./models"}
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
# Создаем путь в стиле Hugging Face Cache
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

echo "📥 Конфигурационные файлы..."
# Список файлов based на предоставленной вами структуре
FILES=(
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "chat_template.json"
    "preprocessor_config.json"
    "video_preprocessor_config.json"
    "model.safetensors.index.json" # Индексный файл нужен для поиска имен весов
)

for file in "${FILES[@]}"; do
    echo "   - $file"
    wget -q --show-progress --continue "https://huggingface.co/$MODEL_ID/resolve/main/$file"
done

echo "📋 Парсим реальную структуру весов..."
SHARDS=$(python3 -c "
import json
with open('model.safetensors.index.json') as f:
    idx = json.load(f)['weight_map']
shards = sorted(set(w for w in idx.values() if 'model-000' in w))
for shard in shards: print(shard)
")

echo "🚀 Скачиваем файлы весов (shards):"
echo "$SHARDS"

# Скачиваем каждый shard по отдельности
echo "$SHARDS" | while IFS= read -r shard; do
    if [[ -n "$shard" ]]; then
        echo "📦 Скачиваем $shard"
        wget --show-progress --continue "https://huggingface.co/$MODEL_ID/resolve/main/$shard"
    fi
done

echo -e "\n✅ Проверка:"
echo "Файлов весов: $(ls *.safetensors 2>/dev/null | wc -l)"
du -sh *.safetensors 2>/dev/null | head -5
echo "🎉 Готово! Используйте:"
echo "model_path = \"$SNAPSHOT_DIR\""