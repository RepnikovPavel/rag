"""
<class 'transformers.models.qwen3.modeling_qwen3.Qwen3Model'>
input_modalities='text'
main_input_name='input_ids'
checkpoint_files=['/mnt/nvme/huggingface/models--Qwen--Qwen3-Embedding-8B/snapshots/main/model-00001-of-00004.safetensors', '/mnt/nvme/huggingface/models--Qwen--Qwen3-Embedding-8B/snapshots/main/model-00002-of-00004.safetensors', '/mnt/nvme/huggingface/models--Qwen--Qwen3-Embedding-8B/snapshots/main/model-00003-of-00004.safetensors', '/mnt/nvme/huggingface/models--Qwen--Qwen3-Embedding-8B/snapshots/main/model-00004-of-00004.safetensors']
may be dump this files?
checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files
vocab_size=151665
hidden_size=4096
config.num_hidden_layers=36->Qwen3DecoderLayer
@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

loading failed here:

def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    load_config: LoadStateDictConfig,
    tp_plan: dict[str, str] | None,
    disk_offload_index: dict | None = None,
):
self.config._attn_implementation controls attention impl call
_global_mapping = {
    "flash_attention_3": flash_attention_forward,
    "flash_attention_2": flash_attention_forward,
    "flex_attention": flex_attention_forward,
    "sdpa": sdpa_attention_forward, <- this is default for qwen3 code
    "paged|flash_attention_3": paged_attention_forward,
    "paged|flash_attention_2": paged_attention_forward,
    "paged|sdpa": sdpa_attention_paged_forward,
    "paged|eager": eager_paged_attention_forward,
}
sdpa code -> transformers/src/transformers/integrations/sdpa_attention.py

"""

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import time
from tqdm import tqdm


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ckptdir',type=str,required=False,default='/mnt/nvme/huggingface')
    args = argparser.parse_args()
    ckptdir = args.ckptdir
    os.makedirs(ckptdir,exist_ok=True)
    device = 'cuda'
    model_path = f"{ckptdir}/models--Qwen--Qwen3-Embedding-8B/snapshots/main"
    max_length = 8192
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # model = AutoModel.from_pretrained(model_path, local_files_only=True,attn_implementation="flash_attention_2",torch_dtype=torch.float16).eval().to('cuda')
    model = AutoModel.from_pretrained(model_path, local_files_only=True,attn_implementation="flash_attention_2").eval().to('cpu')
    # model = AutoModel.from_pretrained(model_path, local_files_only=True,dtype=torch.float16).eval().to('cuda')

    print('type(model):',type(model))
    print('switch to float16')
    model = model.to(torch.float16)
    
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "The capital of China is Beijing."
    ]
    input_texts = queries + documents

    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    
    # first calls
    with torch.no_grad():
        input_ids= batch_dict['input_ids']
        attention_mask =batch_dict['attention_mask']
        n_sequences=batch_dict.n_sequences
        encodings=batch_dict.encodings
        # outputs = model(**batch_dict)
        outputs = model.iter_forward_gpu_buff(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            num_layers_in_buffer=18
        )
        # print('embeddings.size()',embeddings.size())
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        print('embeddings.size()',embeddings.size())
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
    
    model.to('cpu')
    N_runs = 8
    for run_idx in tqdm(range(N_runs),desc='perf test num_layers_in_buffer=18'):
        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            input_ids= batch_dict['input_ids']
            attention_mask =batch_dict['attention_mask']
            n_sequences=batch_dict.n_sequences
            encodings=batch_dict.encodings
            # outputs = model(**batch_dict)
            outputs = model.iter_forward_gpu_buff(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                num_layers_in_buffer=18
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter_ns()
            print(f'e2e:{(t2-t1)/1e6:.1f} ms\n')
            # print('embeddings.size()',embeddings.size())
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            print('embeddings.size()',embeddings.size())
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:2] @ embeddings[2:].T)
        
    print(scores.tolist())

    # model.to('cpu')
    N_runs = 8
    for run_idx in tqdm(range(N_runs),desc='perf test num_layers_in_buffer=8'):
        with torch.no_grad():
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            input_ids= batch_dict['input_ids']
            attention_mask =batch_dict['attention_mask']
            n_sequences=batch_dict.n_sequences
            encodings=batch_dict.encodings
            # outputs = model(**batch_dict)
            outputs = model.iter_forward_gpu_buff(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                num_layers_in_buffer=8
            )
            torch.cuda.synchronize()
            t2 = time.perf_counter_ns()
            print(f'e2e:{(t2-t1)/1e6:.1f} ms\n')
            # print('embeddings.size()',embeddings.size())
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            print('embeddings.size()',embeddings.size())
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:2] @ embeddings[2:].T)
        
    print(scores.tolist())
    