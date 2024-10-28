'''
Author: Xiaocong Yang
LastEditors: Xiaocong Yang
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, pipeline
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
    
    
class Retriever():
    def _encode_corpus(self, corpus):
        raise NotImplementedError
    
    def encode_query(self, query):
        raise NotImplementedError
    
    def retrieve(self, queries, topk):
        raise NotImplementedError
    
## to do: implement BM25Retriever
# class BM25Retriever(Retriever):
    
class DenseRetriever(Retriever):
    def __init__(self, retriever_trained=True, model_modified=True):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
        self.retriever_trained = retriever_trained
        if retriever_trained:
            # self.model = AutoModel.from_pretrained("/shared/xiaocong/graph_rag/bge-large-zh-v1.5_math23k_10epoch_v5_masked.pt").cuda()
            self.model = torch.load("/shared/xiaocong/graph_rag/bge-large-zh-v1.5_math23k_6epoch_v6_masked.pt")
            # print(self.model)
            # self.model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').cuda()
            # # load checkpoint
            # self.model.load_state_dict(torch.load("/shared/xiaocong/graph_rag/bge-large-zh-v1.5_math23k_10epoch_v5_masked.pt"))
        else:
            self.model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
        self.model_modified = model_modified
        self.corpus_embeddings = torch.load("tal_corpus_embeddings_trained_v1_Chinese.pt").to("cuda:0")
        self.corpus = load_dataset("json",data_files="/shared/xiaocong/TAL-SCQ5K/ch_single_choice_constructed_5K/ch_single_choice_train_3K.jsonl", split="train")
        # self.corpus_embeddings, self.corpus = self._encode_corpus("tal")
        # print("Corpus embeddings shape: ", self.corpus_embeddings.shape)
        # print("Corpus length: ", len(self.corpus))
        # assert len(self.corpus_embeddings) == len(self.corpus) == 3000
        # # # # save corpus embeddings
        # torch.save(self.corpus_embeddings, "tal_corpus_embeddings_trained_v1_Chinese.pt")
        # # # save corpus
        # with open("ape210k_corpus_trained_v1_Chinese.json", "w") as f:
        #     f.write(json.dumps(self.corpus, ensure_ascii=False))
        # if not self.retriever_trained:
        #     self.corpus_embeddings = torch.load("ape210k_corpus_embeddings_untrained_v1_Chinese.pt").to("cuda:0")
        #     with open("ape210k_corpus_untrained_v1_Chinese.json", "r") as f:
        #         self.corpus = json.load(f)
        # else:
        #     self.corpus_embeddings = torch.load("ape210k_corpus_embeddings_trained_v1_Chinese.pt").to("cuda:4")
        #     with open("ape210k_corpus_trained_v1_Chinese.json", "r") as f:
        #         self.corpus = json.load(f)
        # self.corpus_embeddings = torch.load("gsm8k_corpus_embeddings_trained.pt")
        # self.corpus_embeddings = torch.load("gsm8k_corpus_embeddings_untrained.pt")
        # self.corpus_embeddings = torch.load("math23k_corpus_embeddings_trained_v6_Chinese.pt")
        
    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _encode_corpus(self, corpus):
        # accelerator = Accelerator()
        if corpus == "tal":
            # corpus = load_dataset("json",data_files="/shared/xiaocong/ape210k/data/train.ape.jsonl", split="train")
            corpus = load_dataset("json",data_files="/shared/xiaocong/TAL-SCQ5K/ch_single_choice_constructed_5K/ch_single_choice_train_3K.jsonl", split="train")
        else:
            corpus = load_dataset(corpus, split="train")
        all_embeddings = []
        # all_data = {"original_text": [], "equation": [], "answer": []}
        dataloader = torch.utils.data.DataLoader(corpus, batch_size=64, shuffle=False, collate_fn=collate_fn)
        # _model, dataloader = accelerator.prepare(self.model, dataloader)
        _model = self.model.to("cuda:0")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                questions = batch["question"]
                inputs = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(_model.bert.device) for k, v in inputs.items()}
                if self.model_modified:
                    embeddings = _model(inputs)
                else:
                # embeddings = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    outputs = _model(**inputs)
                    embeddings = outputs.pooler_output
                # gather all embeddings and data
                # embeddings = accelerator.gather(embeddings)
                all_embeddings.append(embeddings)
                # all_data["original_text"].extend(accelerate.utils.gather_object(batch["original_text"]))
                # all_data["equation"].extend(accelerate.utils.gather_object(batch["equation"]))
                # all_data["answer"].extend(accelerate.utils.gather_object(batch["answer"]))
        all_embeddings = torch.cat(all_embeddings)
        return all_embeddings, corpus
    
    def encode_query(self, query):
        self.model = self.model.to("cuda:0")
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        with torch.no_grad():
            # outputs = self.model(**inputs)
            # embeddings = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            # embeddings = outputs.pooler_output
            if self.model_modified:
                embeddings = self.model(inputs)
            else:
                outputs = self.model(**inputs)
                embeddings = outputs.pooler_output
        return embeddings

    def retrieve(self, queries, topk=8):
        # queries = queries["original_text"]
        query_embeddings = self.encode_query(queries)
        # normalize embeddings
        corpus_embeddings = F.normalize(self.corpus_embeddings, p=2, dim=-1)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        # compute similarity
        similarity_matrix = torch.matmul(query_embeddings, corpus_embeddings.T)
        # get topk indices for each query
        topk_indices = torch.topk(similarity_matrix, topk).indices
        return topk_indices
    

class SymbolicRetriever(Retriever):
    def __init__(self, corpus):
        self.corpus = self._encode_corpus(corpus)
    
    def _similarity(self, a, b, score_cutoff):
        return Levenshtein.ratio(a, b, score_cutoff=score_cutoff)
    
    def _encode_corpus(self, corpus):
        return ["".join(item) for item in corpus["target_template"]]
    
    def encode_query(self, query):
        return query["target_template"]
    
    def retrieve(self, queries, topk=8):
        queries = self.encode_query(queries)
        similarity_matrix = torch.zeros(len(queries), len(self.corpus))
        for i, query in enumerate(queries):
            for j, corpus in enumerate(self.corpus):
                similarity_matrix[i, j] = self._similarity(query, corpus, 0)
        topk_indices = torch.topk(similarity_matrix, topk).indices
        return topk_indices

def collate_fn(batch):
    questions = [item["problem"] + "\nOptions: " + str(item["answer_option_list"]) for item in batch]
    answers = [item["answer_analysis"][0] + "#### Answer: " + str(item["answer_value"]) for item in batch]
    return {"question": questions, "answer": answers}

# def collate_fn(batch):
#     original_questions = [item['original_text'] for item in batch]
#     computational_graphs = ["".join(item['target_template']) for item in batch]
#     equations = [item['equation'] for item in batch]
#     return {"original_text": original_questions, "target_template": computational_graphs, "equation": equations}

# def collate_fn(batch):
#     original_questions = [item['original_text'] for item in batch]
#     equations = [item['equation'] for item in batch]
#     answers = [item['ans'] for item in batch]
#     return {"original_text": original_questions, "equation": equations, "answer": answers}

if __name__ == "__main__":
    retriever = DenseRetriever(retriever_trained=True, model_modified=True)
    # generator = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', torch_dtype=torch.bfloat16).to("cuda:3")
    pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",torch_dtype=torch.bfloat16,device="cuda:0")
    
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')
    # tokenizer.pad_token = tokenizer.eos_token
    # train_dataset = load_dataset("Gxg/Math23K", split="train")
    # train_dataset = load_dataset("json",data_files="/shared/xiaocong/ape210k/data/train.ape.jsonl", split="train")
    # val_dataset = load_dataset("json",data_files="/shared/xiaocong/ape210k/data/valid.ape.jsonl", split="train")
    val_dataset = load_dataset("json",data_files="/shared/xiaocong/TAL-SCQ5K/ch_single_choice_constructed_5K/ch_single_choice_test_2K.jsonl", split="train")
    # retriever = SymbolicRetriever(train_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    with open("tal_val_results_trained_8_shots.jsonl", "w") as f:
        for batch in tqdm(val_dataloader):
            prompts = []
            # questions = batch["original_text"]
            retrieved_indices = retriever.retrieve(batch["question"])
            for i in range(retrieved_indices.shape[0]):
                prompt = [{"role":"system", "content":"Answer the question following the format of given examples."}]
                for j in range(retrieved_indices.shape[1]):
                    prompt.append({"role": "user", "content": retriever.corpus["problem"][retrieved_indices[i, j]] + "\n选项: " + str(retriever.corpus["answer_option_list"][retrieved_indices[i, j]])})
                    prompt.append({"role": "assistant", "content": retriever.corpus["answer_analysis"][retrieved_indices[i, j]][0] + "#### 答案: " + str(retriever.corpus["answer_value"][retrieved_indices[i, j]])})
                # random sample 8 examples from the training dataset
                # for _ in range(8):
                #     random_index = torch.randint(0, len(train_dataset), (1,)).item()
                #     prompt.append({"role": "user", "content": train_dataset["original_text"][random_index]})
                #     prompt.append({"role": "assistant", "content": train_dataset["equation"][random_index]})
                prompt.append({"role": "user", "content": batch["question"][i]})
                prompts.append(prompt)
            outputs = pipeline(prompts, max_new_tokens=512)
            for i, output in enumerate(outputs):
                line = {"answer": batch["answer"][i], "generated": output[0]["generated_text"][-1]["content"]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    # with open("gsm8k_test_results_zero_shot.jsonl", "w") as f:
    #     with torch.no_grad():
    #         for batch in tqdm(test_dataloader):
    #             queries = batch["question"][0]
    #             answers = batch["answer"][0]
    #             topk_indices = retriever.retrieve(queries).squeeze()
    #             prompts = []
    #             # for i in range(topk_indices.shape[0]):
    #             #     new_lines = [{"role":"user", "content":train_dataset["question"][topk_indices[i]]}, 
    #             #                  {"role":"assistant", "content":train_dataset["answer"][topk_indices[i]]}]
    #             #     prompts.extend(new_lines)
    #             # # add the query to the prompts
    #             prompts.extend([{"role":"user", "content":queries + " The final answer should be prefixed by ####."}])
    #             ## few-shot generation
    #             # inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    #             ## zero-shot generation
    #             # inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    #             # inputs = {k: v.to("cuda:3") for k, v in inputs.items()}
    #             outputs = pipeline(prompts, max_new_tokens=512)
    #             generated_text = outputs[0]["generated_text"][-1]
    #             line = {"answer": answers, "generated": generated_text["content"]}
    #             f.write(json.dumps(line) + "\n")
    #             # output the generated part only
    #             # generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
    #             # output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #             # all_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #             # for i, output_text in enumerate(output_texts):
    #             #     line = {"answer": answers[i], "generated": output_text, "all": all_texts[i]}
    #             #     f.write(json.dumps(line) + "\n")