import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from transformers import BertForSequenceClassification

from szx82.models.model_env import MODEL as ModelEnv

"""Model shell for masked words
"""

class MODEL(ModelEnv):
    FILE_PREFIX = 'BERT_CLS'

    def __init__(self, config_or_path, shell=None, **kwargs):
        # `config_or_model` can be a restored `Model` object coming with 
        # its oun config object, or it can be a configuration object to be used 
        # to create a `Model` object
        super().__init__(
            BertForSequenceClassification, config_or_path, shell)
    
    def set_criterion(self, weight):
        pass

    def final_adj(self):
        pass
    # Expected input batch_size (16) to match target batch_size (64).
    def forward(self, batch):
        model_out = self.model(
                output_hidden_states=True, 
                return_dict=True,
                **batch['input']
            )

        logits = model_out.logits 
        loss = model_out.loss
        # import pdb; pdb.set_trace()
        
        self.current = {
            'loss': loss.cpu().detach().tolist(),
            'pred': torch.topk(
                            logits.cpu().detach(), k=1)[1].flatten().tolist(),
            'true': batch['input']['labels'].flatten().tolist()
        }

        return loss
    
    def accuracy(self, current_cumulated):
        y_true = np.array(sum(current_cumulated['true'], []))
        # y_true = np.array([item for sublist 
        #                    in current_cumulated['true'] for item in sublist])
        y_pred = np.array(sum(current_cumulated['pred'], []))
        acc = sum(y_true == y_pred) / len(y_true)
        # cm = confusion_matrix(
        #     y_true=y_true, 
        #     y_pred=y_pred, normalize='all')
        msg = f'accuracy:{acc:.2f}'
        # return acc, msg
        msg = f'acc:{acc:.2f}'
        return {
            'acc': {'acc': acc}, 
            'msg': msg, 
            'accuracy': acc}

 