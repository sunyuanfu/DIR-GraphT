import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from utils import init_random_state, get_available_devices
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
from tqdm.autonotebook import trange

LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120}


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        init_random_state(seed)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
    

class LLMSentenceTextEncoder(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(self, llm_name, cache_dir="cache_data/model", max_length=500):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'


    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        model.config.use_cache = False
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]
        
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="cache_data/model", batch_size=1, multi_gpu=False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMSentenceTextEncoder(llm_name, cache_dir=cache_dir)
        self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()
