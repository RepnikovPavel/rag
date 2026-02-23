import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import argparse
import os

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
    argparser.add_argument('--ckptdir', type=str, required=False, default='/mnt/nvme/huggingface')
    args = argparser.parse_args()
    ckptdir = args.ckptdir
    os.makedirs(ckptdir, exist_ok=True)
    
    device = 'cpu'
    model_path = f"{ckptdir}/models--Qwen--Qwen3-Embedding-8B/snapshots/main"
    max_length = 8192
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Загружаем модель сразу в float16 и на нужный девайс для экономии памяти
    model = AutoModel.from_pretrained(
        model_path, 
        local_files_only=True, 
        torch_dtype=torch.float16
    ).eval().to(device)

    print('type(model):', type(model))

    task = 'Given a web search query, retrieve relevant passages that answer the query'

    # 1. Подготовка данных
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    
    documents = [
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        "The capital of China is Beijing."
    ]

    # ---------------------------------------------------------
    # ЭТАП 1: Обработка документов (Documents)
    # ---------------------------------------------------------
    print("\nProcessing documents...")
    batch_docs = tokenizer(
        documents,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Используем стандартный вызов модели
        outputs_docs = model(**batch_docs)
        # Пулинг и нормализация
        doc_embeddings = last_token_pool(outputs_docs.last_hidden_state, batch_docs['attention_mask'])
        doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    print(f"Doc embeddings shape: {doc_embeddings.shape}")

    # ---------------------------------------------------------
    # ЭТАП 2: Обработка запросов (Queries)
    # ---------------------------------------------------------
    print("\nProcessing queries...")
    batch_queries = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs_queries = model(**batch_queries)
        # Пулинг и нормализация
        query_embeddings = last_token_pool(outputs_queries.last_hidden_state, batch_queries['attention_mask'])
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    print(f"Query embeddings shape: {query_embeddings.shape}")

    # ---------------------------------------------------------
    # ЭТАП 3: Вычисление скоров (Similarity Scores)
    # ---------------------------------------------------------
    # Матрица скоров: (количество запросов) x (количество документов)
    # queries[:2] -> query_embeddings
    # documents[2:] -> doc_embeddings
    scores = query_embeddings @ doc_embeddings.T

    print("\nFinal Scores:")
    print(scores.tolist())
    
    # Ожидаемый результат (значения должны совпасть с точностью до float):
    # [[0.07525634765625, 0.74951171875], [0.6318359375, 0.0880126953125]]