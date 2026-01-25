#!/bin/bash
CKPTDIR=${1:-"./models"}
MODEL_ID="Qwen/Qwen3-Reranker-8B"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-Reranker-8B/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

echo "📥 Конфигурационные файлы..."
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/config.json"
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/tokenizer.json" 
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/tokenizer_config.json"
wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/special_tokens_map.json"

echo "📋 Парсим реальную структуру весов..."
SHARDS=$(python3 -c "
import json
with open('model.safetensors.index.json') as f:
    idx = json.load(f)['weight_map']
shards = sorted(set(w for w in idx.values() if 'model-000' in w))
for shard in shards: print(shard)
")

echo "🚀 Скачиваем 5 safetensors файлов:"
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
