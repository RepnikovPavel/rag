```sh
bash test/qwen3/download_embedding_model.sh /mnt/nvme/transformers_ckpts
bash test/qwen3/download_reranker_model.sh /mnt/nvme/transformers_ckpts
```


```sh
# HF_HUB_ENABLE_HF_TRANSFER=1 python3 test/qwen3/demo.py --ckptdir=/mnt/nvme/transformers_ckpts
python3 test/qwen3/demo_embeddings.py --ckptdir=/mnt/nvme/transformers_ckpts
```

```sh
python3 test/qwen3/demo_reranker.py --ckptdir=/mnt/nvme/transformers_ckpts
```