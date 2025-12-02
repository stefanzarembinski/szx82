import numpy as np
import torch
# https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/bert#transformers.BertForPreTraining
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertForPreTraining, BertConfig

from szx82.models.model_env import MODEL as ModelEnv

"""Model shell for masked words
"""

class Config(BertConfig):  
    def __init__(
            self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None, 
            device='cpu',
            vocab=None,
            pretrained_path=None,
            **kwargs
            ):
        super().__init__(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs
            )
        self.vocab = vocab
        self.device = device
        self.pretrained_path = pretrained_path

class MODEL(ModelEnv):
    FILE_PREFIX = 'BERT_PRE'

    def __init__(self, config_or_path, shell=None, **kwargs):
        # `config_or_model` can be a restored `Model` object coming with 
        # its oun config object, or it can be a configuration object to be used 
        # to create a `Model` object
        super().__init__(BertForPreTraining, config_or_path, shell)

    def final_adj(self):
        pass

    def loss(self, input, model_out, reduction='mean', ):
        vocab_size = self.model.config.vocab_size
        loss_fct = CrossEntropyLoss(reduction=reduction)
        loss = None
        if 'labels' in input:
            labels = input['labels']
            prediction_scores = model_out.prediction_logits
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, vocab_size), labels.view(-1))
            loss = masked_lm_loss
                 
        if 'next_sentence_label' in input:
            next_sentence_label = input['next_sentence_label']
            seq_relationship_score = model_out.seq_relationship_logits
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), 
                next_sentence_label.view(-1))
            if loss is None:
                loss = next_sentence_loss
            else:
                loss += next_sentence_loss

        return loss

    def forward(self, batch):
        model_out = self.model(
                output_hidden_states=True, 
                return_dict=True,
                **batch['input']
            )
        prediction_logits = model_out.prediction_logits
        seq_relationship_logits = model_out.seq_relationship_logits

        # Total loss as the sum of the masked language modeling loss and the 
        # next sequence prediction (classification) loss:
        loss = model_out.loss
        if loss is None:
            loss = self.loss(batch['input'], model_out=model_out)

        mlm_count = None
        masked_pred = None
        mlm_eq = None
        if 'labels' in batch['input']:
            labels = batch['input']['labels']
            masked_ids = (labels != -100).nonzero(as_tuple=True)
            masked_lbs = labels[masked_ids]
            mlm_count = len(masked_lbs)
            pred = torch.topk(prediction_logits, k=1, dim=-1)[1].squeeze() 
            masked_pred = pred[masked_ids]
            mlm_eq = sum(masked_lbs == masked_pred).item()
        
        # batch result is collected for accuracy calculations
        self.current = {'loss': loss.cpu().detach().tolist(),}
        if seq_relationship_logits is not None:
            self.current.update({
                    'nsp_eq': sum(
                torch.topk(seq_relationship_logits, k=1)[1] \
                    == batch['input']['next_sentence_label']).item(),
                    'nsp_count': len(batch['input']['next_sentence_label']),
        })
        if mlm_eq is not None:
            self.current.update({
            'mlm_eq': mlm_eq,
            'mlm_count': mlm_count
            })
        return loss
    
    def accuracy(self, current_cumulated):
        if 'nsp_eq' in current_cumulated:
            nsp = sum(current_cumulated['nsp_eq']) \
                        / sum(current_cumulated['nsp_count'])
        else:
            nsp = None
        
        if 'mlm_eq' in current_cumulated:
            mlm = sum(current_cumulated['mlm_eq']) \
                                    / sum(current_cumulated['mlm_count'])
        else:
            mlm = None
        
        if (nsp is not None) and (mlm is not None):
            return {
                'acc': {'nsp': nsp, 'mlm': mlm},  
                'msg': f'nsp,mlm:{nsp:.2f},{mlm:.2f}', 
                'accuracy': nsp + mlm}
        if nsp is not None:
            return {
                'acc': {'nsp': nsp}, 
                'msg': f'nsp:{nsp:.2f}', 
                'accuracy': nsp}
        if mlm is not None:
            return {
                'acc': {'mlm': mlm}, 
                'msg': f'mlm:{mlm:.2f}', 
                'accuracy': mlm}