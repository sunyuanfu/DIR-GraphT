import os
import json
import torch
import csv
from .dgl_dataset import CustomDGLDataset


def load_gpt_preds(dataset, topk):
    preds = []
    fn = f'gpt_preds/{dataset}.csv'
    print(f"Loading topk preds from {fn}")
    with open(fn, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            inner_list = []
            for value in row:
                inner_list.append(int(value))
            preds.append(inner_list)

    pl = torch.zeros(len(preds), topk, dtype=torch.long)
    for i, pred in enumerate(preds):
        pl[i][:len(pred)] = torch.tensor(pred[:topk], dtype=torch.long)+1
    return pl


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0):
    if dataset == 'cora':
        from .load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from .load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from .load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'ogbn-products':
        from .load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv-2023':
        from .load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    elif dataset == 'chemhiv':
        from .load_chem import get_raw_text_hiv as get_raw_text
        num_classes = 1
    elif dataset == 'chempcba':
        from .load_chem import get_raw_text_hiv as get_raw_text
        num_classes = 128
    elif dataset == 'ogbg-pcba':
        from .load_ogbpcba import get_raw_text_ogbpcba as get_raw_text
        num_classes = 128
    elif dataset == 'ogbg-ppa':
        from .load_ogbppa import get_raw_text_ppa as get_raw_text
        num_classes = 37
    else:
        exit(f'Error: Dataset {dataset} not supported')

    # for training GNN
    if not use_text:
        if dataset == "chempcba":
            data, _ = get_raw_text(use_text=False, seed=seed, name="chempcba")
        else:
            data, _ = get_raw_text(use_text=False, seed=seed)
        if use_dgl:
            data = CustomDGLDataset(dataset, data)
        return data, num_classes

    # for finetuning LM
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = 'gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed)

    return data, num_classes, text
