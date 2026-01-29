pip install requests beautifulsoup4
pip install "transformers>=4.57.0"


```
brew install huggingface-cli
pip install hf_transfer
sudo mkdir -p /mnt/nvme/huggingface
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct \
  --include "*safetensors*" "*model*" "config.json" \
  --local-dir /mnt/nvme/huggingface/qwen3-vl-weights \
  --resume-download
```

```
sudo mkdir -p /mnt/nvme/huggingface/qwen2.5-72b-instruct 
sudo chown -R $(whoami):$(whoami) /mnt/nvme/huggingface/qwen2.5-72b-instruct 
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
  --include "*safetensors*" "*model*" "config.json" \
  --local-dir /mnt/nvme/huggingface/qwen2.5-72b-instruct \
  --resume-download
```


```
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download meta-llama/Llama-3.2-90B-Vision-Instruct \
  --include "*safetensors*" "*model*" "config.json" \
  --local-dir ./llama3.2-90b-vision \
  --resume-download
```