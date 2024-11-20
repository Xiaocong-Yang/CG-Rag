## Training code
Line 27: The NUM_POSITIVES MUST be set to 1. Don't change it.

Line 70: Change the base model to the English version of BGE. (BAAI/bge-large-en-v1.5)

Line 83: We added two MLP layers to the bge model. 

Line 91: Change the dataset to its english version. The process code is as following:
```
from datasets import load_dataset
import json
from tqdm import trange

raw_dataset = load_dataset("Gxg/Math23K", split="train")
all_computational_graphs = raw_dataset['target_template']
with open("processed_math23k.jsonl",'w') as f:
    for i in trange(len(raw_dataset)):
        ## find all data with the same computational graph
        current_computational_graph = all_computational_graphs[i]
        positive_indices = [j for j, graph in enumerate(all_computational_graphs) if graph == current_computational_graph]
        # remove the current data from the positive indices
        positive_indices.remove(i)
        if len(positive_indices) > 0:
            line = {"original_text":raw_dataset[i]['original_text'],"positive_texts":[raw_dataset[j]['original_text'] for j in positive_indices],"target_template":current_computational_graph, "positive_indices":positive_indices}
            f.write(json.dumps(line, ensure_ascii=False)+'\n')
```

Line 98: Increase the training epochs to ~6 

Line 152: Change the saving path

## Testing code
Use rag_testing_{task_name}.py scripts.

Use --random flag for random shots experiments. Use --retriever_trained and --model_modified for retrieval with trained model experiments. The trained retriever can be found here: https://huggingface.co/xiaocong01/BGE-large-en-v1.5-Computational-Graph (English) and https://huggingface.co/xiaocong01/BGE-large-zh-v1.5-Computational-Graph (Chinese). The data for ape210k can be found at https://github.com/Chenny0808/ape210k/tree/master/data.

