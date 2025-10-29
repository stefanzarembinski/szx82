import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification, BertConfig
from sklearn.metrics import confusion_matrix

import szx81.core.core_ as co
from szx81.models.model_env import MODEL as ModelEnv
from szx81.models.bert_rv.dataset import Data

"""Model shell for masked words
"""

class Config(BertConfig):
    def __init__(
            self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None, 
            path=None,
            device='cpu',
            criterion=CrossEntropyLoss,
            crit_weight=None,
            **kwargs
            ):
        super().__init__(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs
            )
        self.path = path
        self.device = device
        self.criterion = criterion
        self.crit_weight=crit_weight

class Model(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.path is not None:
            self = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=config.path,
                config = config,
                cache_dir=None,
                local_files_only=True,
            )

class MODEL(ModelEnv):
    DATASET = (Data.CLASS_LABELS)
    FILE_PREFIX = 'BERT_CLS'

    def __init__(self, config, device='cpu', shell=None, 
                 *args, **kwargs):
        super().__init__(Model, config, device, shell, *args, **kwargs)
        self.y = None
        self.acc = None

    def set_criterion(self, weight):
        self.config.crit_weight = weight
        self.criterion = self.config.criterion(weight=weight, reduction='none')

    def final_adj(self):
        self.tokenizer = self.shell.transformer_shell.data_object.tokenizer

    def forward(self, batch):
        model_out = self.model(
                output_hidden_states=True, 
                return_dict=True,
                **batch['input']
            )
        
        logits = model_out.logits
        loss = model_out.loss
        # import pdb; pdb.set_trace()
        
        # Single-label classification is a type of machine learning task where each input (e.g., text, image, or other data) is assigned exactly one label from a predefined set of categories.

        # # From BertForSequenceClassification source code:
        # loss = None
        # labels = batch['input']['labels']
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # match = model_out.logits.detach().topk(1)[1].transpose(
        #     0, 1)[-2][0] == batch['input']['input_ids'].T[-2]

        # self.acc = (match.sum() / len(match.flatten())).item()
        
        self.current = {
            'loss': loss.cpu().detach().tolist(),
            'pred': torch.topk(
                            logits.cpu().detach(), k=1)[1].flatten().tolist(),
            'true': batch['input']['labels'].flatten().tolist()
        }

        return logits, loss
    
    def accuracy(self, current_cumulated):
        y_true = np.array(current_cumulated['true']).flatten()
        y_pred = np.array(current_cumulated['pred']).flatten()
        acc = sum(y_true == y_pred) / len(y_true)
        cm = confusion_matrix(
            y_true=y_true, 
            y_pred=y_pred, normalize='all')
        msg = f'accuracy:{acc:.2f}'
        return acc, msg

 