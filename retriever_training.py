'''
Author: Xiaocong Yang
LastEditors: Xiaocong Yang
'''
import os
import sys
import Levenshtein
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

from torch import Tensor
import torch.nn as nn

from datasets import load_dataset
import random

from accelerate import Accelerator

import os



NUM_POSITIVES = 1

def similarity(a, b, score_cutoff):
    return Levenshtein.ratio(a, b, score_cutoff=score_cutoff)

def batch_similarity(computational_graph_batch, score_cutoff=0.8):
    # calculate the similarity between any two elements in the batch
    n = len(computational_graph_batch)
    similarity_matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = similarity(computational_graph_batch[i], computational_graph_batch[j], score_cutoff)
    return similarity_matrix


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# def collate_fn(batch):
#     questions = [item['question'] for item in batch]
#     computational_graphs = ["".join(item['computational_graph']) for item in batch]
#     return {"question": questions, "computational_graph": computational_graphs}

# def collate_fn(batch):
#     original_questions = [item['original_question'] for item in batch]
#     modified_questions = [item['modified_question'] for item in batch]
#     return original_questions + modified_questions

def collate_fn(batch):
    outputs = []
    for data in batch:
        outputs.append(data["original_text"])
        positive_samples = random.choices(data["positive_texts"], k=NUM_POSITIVES)
        outputs.extend(positive_samples)
    assert len(outputs) == (NUM_POSITIVES + 1) * len(batch)
    return outputs

class ModifiedModel(nn.Module):
    def __init__(self):
        super(ModifiedModel, self).__init__()
        self.bert = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5', torch_dtype=torch.bfloat16)
        self.extra_pooler = torch.nn.Sequential(torch.nn.Linear(1024, 1024, dtype=torch.bfloat16), torch.nn.ReLU(), torch.nn.Linear(1024, 1024, dtype=torch.bfloat16))
    
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.extra_pooler(embeddings)
        return embeddings
    
if __name__ == "__main__":
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    # model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5', torch_dtype=torch.bfloat16)
    # model = ModifiedModel()
    model = torch.load("/shared/xiaocong/graph_rag/bge-large-zh-v1.5_math23k_5epoch_v6_masked.pt")
    # print all name of model parameters
    # for name, param in model.named_parameters():
    #     print(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # dataset = load_dataset('json', data_files="/shared/xiaocong/math23k_english_4ksubset.jsonl", split='train')
    # dataset = load_dataset("csv", data_files="/shared/xiaocong/gsm8k_modified.csv", split="train")
    # dataset = load_dataset("Gxg/Math23K", split="train")
    dataset = load_dataset("json", data_files="/shared/xiaocong/graph_rag/processed_math23k.jsonl", split="train")
    ## remove the samples with more than 10 positive samples to avoid false positive
    selected_dataset = [item for item in dataset if len(item["positive_texts"]) < 20]
    dataloader = torch.utils.data.DataLoader(selected_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    total_loss = 0
    total_steps = 0
    for epoch in range(1):
        for batch in tqdm(dataloader):
            total_steps += 1
            # calculate predicted similarity score
            # inputs = tokenizer(batch['original_text'], padding=True, truncation=True, return_tensors='pt')
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            embeddings = model(inputs)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            # get embeddings from different devices
            embeddings = embeddings.contiguous()
            all_embeddings = [torch.empty_like(embeddings) for _ in range(accelerator.num_processes)]
            torch.distributed.all_gather(all_embeddings, embeddings)
            all_embeddings[accelerator.process_index] = embeddings
            embeddings = torch.cat(all_embeddings, dim=0)
            predicted_scores = embeddings @ embeddings.T
            # mask the diagonal elements
            predicted_scores = predicted_scores - 1e16 * torch.eye(embeddings.shape[0]).to(predicted_scores.device)
            # apply softmax to the predicted scores as contrastive learning
            # predicted_scores = torch.nn.functional.softmax(predicted_scores / 0.05, dim=1)
            # every consecutive NUM_POSITIVES + 1 samples are labeled as positive samples
            true_scores = torch.zeros(embeddings.shape[0], embeddings.shape[0], dtype=torch.bfloat16).to(predicted_scores.device)
            block_size = NUM_POSITIVES + 1
            for i in range(0, embeddings.shape[0], block_size):
                true_scores[i:i + block_size, i:i + block_size] = 1
            # mask the diagonal elements
            true_scores = true_scores - torch.eye(embeddings.shape[0]).to(predicted_scores.device)
            # calculate true similarity score using Levenshtein distance
            # true_scores = batch_similarity(batch['computational_graph'], score_cutoff=0.85).to("cuda:0")
            # true_scores = torch.eye(len(batch)) + torch.diag(torch.ones(len(batch) // 2), diagonal=len(batch) // 2) + torch.diag(torch.ones(len(batch) // 2), diagonal=-len(batch) // 2)
            # true_scores = true_scores.to("cuda:0")
            # print("true_scores", true_scores)
            # print("predicted_scores", predicted_scores)
            # loss = F.binary_cross_entropy_with_logits(predicted_scores, true_scores, reduction='mean')
            loss = F.cross_entropy(predicted_scores / 0.05, true_scores, reduction='mean')
            # don't consider the diagonal elements
            # mask = torch.ones(embeddings.shape[0], embeddings.shape[0]).to(predicted_scores.device)
            # mask = mask - torch.eye(embeddings.shape[0]).to(predicted_scores.device)
            # loss = loss * mask
            # loss = loss.mean()
            accelerator.backward(loss)
            total_loss += loss.item()
            # update model parameters
            optimizer.step()
            
            if total_steps % 5 == 0 and accelerator.is_main_process:
                print(f"Step {total_steps}: loss {total_loss / 5}")
                total_loss = 0            
                print(predicted_scores[0][:10])
                
        
    # save model checkpoint
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model, "/shared/xiaocong/graph_rag/bge-large-zh-v1.5_math23k_6epoch_v6_masked.pt")