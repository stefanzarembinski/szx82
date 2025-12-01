import os.path as path
from os import path
import torch

from szx82.models.bert.train import Train as TRAIN
import szx82.models.bert.pre.model as PRE
from szx82.models.bert.re_train import data, ReTrain as RE_TRAIN
import szx82.models.bert.cls.model as CLS

DATA_STORE = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store'
DATA_STORE_G = 'G:/My Drive/workplaces/szx81/EURUSD/data_store'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainer():
    training_shell = RE_TRAIN(
        file_name='bert',
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

    training_shell.model = PRE.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,
            'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_pre.pkl'), DEVICE, scale=0.3)
    training_shell.pretrained_path = path.join(
        DATA_STORE, 
        path.join(
            'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;/BERT_PRE_bert_pred3', 'BERT_PRE_bert_pred3_bst_.pt'))
    training_shell.batch_size = 128
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='pre4', force=False)
    training_shell.train()
    
trainer()

def main():
    trainer()

# python -m szx82.models.bert.retrainings
if __name__ == '__main__':
    main()