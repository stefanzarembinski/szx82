import torch
# https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/bert#transformers.BertForPreTraining
from transformers import BertForPreTraining, BertConfig

from szx82.models.model_env import MODEL as ModelEnv

"""Model shell for masked words
"""

def model_factory(config_or_path):
    if isinstance(config_or_path, str):
        return BertForPreTraining.from_pretrained(
                pretrained_model_name_or_path=config_or_path)
    if config_or_path.pretrained_path is not None:
            return BertForPreTraining.from_pretrained( 
                pretrained_model_name_or_path=config_or_path.pretrained_path)
    return BertForPreTraining(config_or_path)

class Config(BertConfig):
    _instance = None  
    def __init__(
            self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None, 
            device='cpu',
            criterion=None,
            crit_weight=None,
            vocab=None,
            pretrained_path=None,
            **kwargs
            ):
        super().__init__(
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs
            )
        self.criterion = criterion
        self.crit_weight = crit_weight
        self.vocab = vocab
        self.device = device
        self.pretrained_path = pretrained_path

    def model_factory(config_or_path):
        pass

class MODEL(ModelEnv):
    DATASET = (BertForPreTraining)
    FILE_PREFIX = 'BERT_PRE'

    def __init__(self, config, shell=None, 
                 *args, **kwargs):
        # `config_or_model` can be a restored `Model` object coming with 
        # its oun config object, or it can be a configuration object to be used 
        # to create a `Model` object
        super().__init__(model_factory, config, shell, *args, **kwargs)

    def set_criterion(self, weight):
        pass

    def final_adj(self):
        pass

    def forward(self, batch):
#         print(f'''
# input_ids: {batch['input']['input_ids'].is_cuda}
# attention_mask: {batch['input']['attention_mask'].is_cuda}
# token_type_ids: {batch['input']['token_type_ids'].is_cuda}
# next_sentence_label: {batch['input']['next_sentence_label'].is_cuda}              
# ''')
#         raise Exception('cuda check')
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

        labels = batch['input']['labels']
        masked_idxs = (labels != -100).nonzero(as_tuple=True)
        masked_lbs = labels[masked_idxs]
        pred = torch.topk(prediction_logits, k=1, dim=-1)[1].squeeze()
        masked_pred = pred[masked_idxs]
        mlm_eq = sum(masked_lbs == masked_pred).item()
        mlm_count = len(masked_lbs)        
        
        # batch result is collected for accuracy calculations
        self.current = {
            'loss': loss.cpu().detach().tolist(),
            'nsp_eq': sum(
                torch.topk(seq_relationship_logits, k=1)[1] \
                    == batch['input']['next_sentence_label']).item(),
            'nsp_count': len(batch['input']['next_sentence_label']),
            'mlm_eq': mlm_eq,
            'mlm_count': mlm_count
        }
        # import pdb; pdb.set_trace()
        return loss
    
    def accuracy(self, current_cumulated):
        nsp = sum(current_cumulated['nsp_eq']) \
                        / sum(current_cumulated['nsp_count']) - 0.5
        mlm = sum(current_cumulated['mlm_eq']) \
                                    / sum(current_cumulated['mlm_count'])
        
        msg = f'nsp,mlm:{nsp:.2f},{mlm:.2f}'
        return {
            'acc': {'nsp': nsp, 'mlm': mlm}, 
            'msg': msg, 
            'accuracy': nsp + mlm}
