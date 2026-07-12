#!/bin/bash
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-30B-A3B-Thinking-2507-FP8/snapshots/main"

echo "STATUS: Creating HF cache structure at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: Failed to create directory"; exit 1; }
cd "$SNAPSHOT_DIR" || { echo "ERROR: Failed to change directory"; exit 1; }

echo "STATUS: Downloading configuration, tokenizer, and index files..."
# Добавлен model.safetensors.index.json
for file in .gitattributes LICENSE README.md config.json generation_config.json \
            model.safetensors.index.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do
    if ! wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"; then
        echo "ERROR: Failed to download $file"
        exit 1
    fi
done

echo "STATUS: Downloading model weights (~31.2GB in 4 shards)..."
# Цикл для скачивания 4 частей весов
for i in 1 2 3 4; do
    # Генерирует имена: model-00001-of-00004.safetensors и т.д.
    WEIGHT_FILE=$(printf "model-%05d-of-00004.safetensors" $i)
    if ! wget --show-progress --continue "https://huggingface.co/$MODEL_ID/resolve/main/$WEIGHT_FILE"; then
        echo "ERROR: Failed to download $WEIGHT_FILE"
        exit 1
    fi
done

echo "STATUS: Verifying file structure..."
# Проверяем количество скачанных шардов
WEIGHT_COUNT=$(ls model-*-of-00004.safetensors 2>/dev/null | wc -l)
# Проверяем 3 конфига (включая индексный файл)
CONFIG_COUNT=$(ls config.json generation_config.json model.safetensors.index.json 2>/dev/null | wc -l)
TOKENIZER_CHECK=$([[ -f tokenizer.json ]] && echo "1" || echo "0")

if [ "$WEIGHT_COUNT" -eq 4 ] && [ "$CONFIG_COUNT" -eq 3 ] && [ "$TOKENIZER_CHECK" -eq 1 ]; then
    echo "VERIFICATION: PASSED"
    echo "WEIGHTS: $WEIGHT_COUNT/4"
    echo "CONFIGS: $CONFIG_COUNT/3"
    echo "TOKENIZER: OK"
    du -sh *.safetensors 2>/dev/null
    echo "STATUS: SUCCESS - Model Qwen3-30B-A3B-Thinking-2507-FP8 is ready"
    exit 0
else
    echo "VERIFICATION: FAILED"
    echo "WEIGHTS: $WEIGHT_COUNT/4"
    echo "CONFIGS: $CONFIG_COUNT/3"
    echo "TOKENIZER: $TOKENIZER_CHECK"
    echo "STATUS: ERROR - Incomplete model structure"
    exit 1
fi