#!/bin/bash
set -e

CKPTDIR=${1:-"/mnt/nvme/huggingface"}
MODEL_ID="Qwen/Qwen3-VL-2B-Thinking-FP8"
SNAPSHOT_DIR="$CKPTDIR/models--Qwen--Qwen3-VL-2B-Thinking-FP8/snapshots/main"

echo "STATUS: Creating HF cache structure at $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: Failed to create directory"; exit 1; }
cd "$SNAPSHOT_DIR" || { echo "ERROR: Failed to change directory"; exit 1; }

echo "STATUS: Downloading configuration and tokenizer files..."
for file in .gitattributes README.md chat_template.json config.json generation_config.json \
            model.safetensors.index.json preprocessor_config.json tokenizer.json \
            tokenizer_config.json video_preprocessor_config.json vocab.json; do
    if ! wget -q --show-progress "https://huggingface.co/$MODEL_ID/resolve/main/$file"; then
        echo "ERROR: Failed to download $file"
        exit 1
    fi
done

echo "STATUS: Downloading model weights (~3.5GB)..."
WEIGHT_FILE="model-00001-of-00001.safetensors"
if ! wget --show-progress --continue "https://huggingface.co/$MODEL_ID/resolve/main/$WEIGHT_FILE"; then
    echo "ERROR: Failed to download $WEIGHT_FILE"
    exit 1
fi

echo "STATUS: Verifying file structure..."
WEIGHT_COUNT=$(ls *.safetensors 2>/dev/null | wc -l)
CONFIG_COUNT=$(ls config.json generation_config.json model.safetensors.index.json preprocessor_config.json video_preprocessor_config.json 2>/dev/null | wc -l)
TOKENIZER_CHECK=$([[ -f tokenizer.json ]] && echo "1" || echo "0")

if [ "$WEIGHT_COUNT" -eq 1 ] && [ "$CONFIG_COUNT" -eq 5 ] && [ "$TOKENIZER_CHECK" -eq 1 ]; then
    echo "VERIFICATION: PASSED"
    echo "WEIGHTS: $WEIGHT_COUNT/1"
    echo "CONFIGS: $CONFIG_COUNT/5"
    echo "TOKENIZER: OK"
    du -sh *.safetensors 2>/dev/null
    echo "STATUS: SUCCESS - Model Qwen3-VL-2B-Thinking-FP8 is ready"
    exit 0
else
    echo "VERIFICATION: FAILED"
    echo "WEIGHTS: $WEIGHT_COUNT/1"
    echo "CONFIGS: $CONFIG_COUNT/5"
    echo "TOKENIZER: $TOKENIZER_CHECK"
    echo "STATUS: ERROR - Incomplete model structure"
    exit 1
fi