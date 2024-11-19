'''
Author: Xiaocong Yang
LastEditors: Xiaocong Yang
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline, AutoModelForCausalLM
import torch
from datasets import load_dataset
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import json
import Levenshtein
from accelerate import Accelerator
import torch.nn as nn
import accelerate
from rank_bm25 import BM25Okapi
import argparse


def collate_fn(batch):
    output = {key:[] for key in batch[0].keys()}
    # convert all values in the batch to strings
    for key in batch[0].keys():
        output[key] = [str(item[key]) for item in batch]
    return output


class ModifiedModel(nn.Module):
    def __init__(self):
        super(ModifiedModel, self).__init__()
        self.bert = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5', torch_dtype=torch.bfloat16)
        self.extra_pooler = torch.nn.Sequential(torch.nn.Linear(1024, 1024, dtype=torch.bfloat16), torch.nn.ReLU(), torch.nn.Linear(1024, 1024, dtype=torch.bfloat16))

    
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.extra_pooler(embeddings)
        return embeddings
    
    
class Retriever():
    def _encode_corpus(self, corpus):
        raise NotImplementedError
    
    def encode_query(self, query):
        raise NotImplementedError
    
    def retrieve(self, queries, topk):
        raise NotImplementedError
    
                
                    
class DenseRetriever(Retriever):
    def __init__(self, retriever_trained=True, model_modified=True, trained_retriever_path=None, corpus_dir=None, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.retriever_trained = retriever_trained
        if retriever_trained:
            self.model = torch.load(trained_retriever_path, map_location="cpu")
        else:
            self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5',torch_dtype=torch.bfloat16)
        self.model_modified = model_modified
        self.device = device
        self.corpus_embeddings, self.corpus = self._encode_corpus(corpus_dir)
    


    def _encode_corpus(self, corpus_path):
        # accelerator = Accelerator()
        corpus = load_dataset(corpus_path, "default", split="train")
        all_embeddings = []
        all_data = {"question": [], "answer":[]}
        dataloader = torch.utils.data.DataLoader(corpus, batch_size=16, shuffle=False, collate_fn=collate_fn)
        # _model, dataloader = accelerator.prepare(self.model, dataloader)
        _model = self.model.to(self.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_texts = batch["question"]
                inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                if self.model_modified:
                    inputs = {k: v.to(_model.bert.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
                if self.model_modified:
                    embeddings = _model(inputs)
                else:
                    outputs = _model(**inputs)
                    embeddings = outputs.pooler_output
                # gather all embeddings and data
                # embeddings = accelerator.gather(embeddings)
                all_embeddings.append(embeddings)
                all_data["question"].extend(batch["question"])
                all_data["answer"].extend(batch["equation"])
        all_embeddings = torch.cat(all_embeddings)
        assert len(all_embeddings) == len(all_data["question"])
        return all_embeddings, all_data
    
    def encode_query(self, query):
        self.model = self.model.to(self.device)
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            if self.model_modified:
                embeddings = self.model(inputs)
            else:
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
        return embeddings

    def retrieve(self, queries, topk=8):
        query_embeddings = self.encode_query(queries)
        # normalize embeddings
        corpus_embeddings = F.normalize(self.corpus_embeddings, p=2, dim=-1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        # compute similarity
        similarity_matrix = torch.matmul(query_embeddings, corpus_embeddings.T)
        # get topk indices for each query
        topk_indices = torch.topk(similarity_matrix, topk).indices
        return topk_indices
    



if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument("--random", action="store_true", help="if use random shots from the training set")
    parse.add_argument("--retriever_trained", action="store_true", help="if the retriever model is trained")
    parse.add_argument("--model_modified", action="store_true", help="if the model is modified")
    parse.add_argument("--topk", type=int, default=8)
    parse.add_argument("--corpus_dir", type=str, default="MU-NLPC/Calc-ape210k")
    parse.add_argument("--output_path", type=str, default="Calc-ape210k_trained_8_shots_llama3B.jsonl")
    parse.add_argument("--trained_retriever_path", type=str, default="/shared/xiaocong/graph_rag/bge-large-en-v1.5_math23k_25%_data_masked.pt")
    parse.add_argument("--generator", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parse.add_argument("--retriever_device", type=str, default="cuda:0")
    parse.add_argument("--generator_device", type=str, default="cuda:1")
    args = parse.parse_args()
    
    if not args.random:
        retriever = DenseRetriever(retriever_trained=args.retriever_trained, model_modified=args.model_modified, trained_retriever_path=args.trained_retriever_path, corpus_dir=args.corpus_dir, device=args.retriever_device)
    train_dataset = load_dataset(args.corpus_dir, "default", split="train")
    val_dataset = load_dataset(args.corpus_dir, "default", split="test")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    pipeline = pipeline("text-generation", model=args.generator, torch_dtype=torch.bfloat16,device=args.generator_device)
    
    for batch in tqdm(val_dataloader):
        with open(args.output_path, "a+") as f:
            prompts = []
            if not args.random:
                retrieved_indices = retriever.retrieve(batch["question"], topk=args.topk)
            for i in range(len(batch["question"])):
                prompt = [{"role":"system", "content":"Answer the question following the format of the given examples."}]
                if not args.random:
                    for j in range(retrieved_indices.shape[1]):
                        prompt.append({"role": "user", "content": retriever.corpus["question"][retrieved_indices[i, j]]})
                        prompt.append({"role": "assistant", "content": retriever.corpus["answer"][retrieved_indices[i, j]]})
                
                else:
                    for _ in range(args.topk):
                        random_index = torch.randint(0, len(train_dataset), (1,)).item()
                        prompt.append({"role": "user", "content": train_dataset["question"][random_index]})
                        prompt.append({"role": "assistant", "content": train_dataset["equation"][random_index]})
                prompt.append({"role": "user", "content": batch["question"][i]})
                prompts.append(prompt)
            outputs = pipeline(prompts, max_new_tokens=512)
            for i, output in enumerate(outputs):
                line = {"answer": batch["equation"][i], "generated": output[0]["generated_text"][-1]["content"]}    
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
