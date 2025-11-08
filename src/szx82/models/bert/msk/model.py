import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, BertConfig

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

class Model(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.path is not None:
            self = BertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=config.path,
                config = config,
                cache_dir=None,
                local_files_only=True,
            )

class MODEL(ModelEnv):
    DATASET = (Data.MASKED)
    FILE_PREFIX = 'BERT_MASK'

    def __init__(self, config, device='cpu', shell=None, 
                 *args, **kwargs):
        super().__init__(Model, config, device, shell, *args, **kwargs)
        self.y = None
        self.acc = None

    def set_criterion(self, weight):
        self.config.crit_weight = weight
        self.criterion = self.config.criterion(weight=weight, reduction='none')

    def final_adj(self):
        self.tokenizer = self.shell.project_shell.data_object.tokenizer

    def forward(self, batch):
        model_out = self.model(
                output_hidden_states=True, 
                return_dict=True,
                **batch['input']
            )
        
        logits = model_out.logits
        loss = model_out.loss

        # import pdb; pdb.set_trace()
        # mask_token_index = (batch['input']['input_ids'] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.MASK)).nonzero(as_tuple=True)[0]
        # predicted_token_id = logits[mask_token_index].argmax(axis=-1)
        # predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_token_id)

        # # From BertForMaskedLM source code:
        # loss_fct = CrossEntropyLoss()  # -100 index = padding token
        # masked_lm_loss = loss_fct(
        #     # 736 == batch_size * self.config.max_position_embeddings
        #     # 261 == self.tokenizer.vocab_size
        #     logits.view(-1, self.config.vocab_size), # shape: [736, 261]
        #     batch['input']['labels'].view(-1)) # shape [736]
        # loss_fct == loss is True

        match = model_out.logits.detach().topk(1)[1].transpose(
            0, 1)[-2][0] == batch['input']['input_ids'].T[-2]

        self.acc = (match.sum() / len(match.flatten())).item()
        
        self.current = {
            'loss': loss.cpu().detach().tolist(),
        }

        return logits, loss
    
    def accuracy(self, current_cumulated):
        value = self.acc
        msg = f'accuracy:{value:.2f}'
        return value, msg

 