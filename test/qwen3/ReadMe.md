```sh
bash download.sh /mnt/nvme/transformers_ckpts
```


```sh
HF_HUB_ENABLE_HF_TRANSFER=1 python3 test/qwen3/demo.py --ckptdir=/mnt/nvme/transformers_ckpts
```