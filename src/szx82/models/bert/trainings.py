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
    training_shell = TRAIN(
        file_name='bert',
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

    """
    training_shell. model = PRE.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_pre.pkl'), DEVICE, scale=0.3)
    training_shell.batch_size = 256
    training_shell.hidden_size = 384 # defaults to 768
    training_shell.intermediate_size = 1536 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 10 # defaults to 12
    training_shell.num_attention_heads = 4 # defaults to 12
    training_shell.dropuot = 0.1 # default is 0.1
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='pred1', force=False )
    training_shell.train()

# training; loss trn,val:1.00,1.00; best:0.53; train:nsp,mlm:0.50,0.01; val.:nsp,mlm:0.50,0.03; ep:2; batch:10/422;
# training; loss trn,val:0.89,0.89; best:0.95; train:nsp,mlm:0.50,0.03; val.:nsp,mlm:0.50,0.03; ep:41; batch:51/422; 
   
    """

    """
    training_shell. model = PRE.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_pre.pkl'), DEVICE, scale=0.3)
    training_shell.batch_size = 128
    training_shell.hidden_size = 384 # defaults to 768
    training_shell.intermediate_size = 1536 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 4 # defaults to 12
    training_shell.num_attention_heads = 4 # defaults to 12
    training_shell.dropuot = 0.1 # default is 0.1
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='pred2', force=False )
    training_shell.train()

# training; loss trn,val:0.51,0.53; best:0.95; train:nsp,mlm:0.76,0.20; val.:nsp,mlm:0.75,0.19; ep:24; batch:19/844;
    fi 2 sep 2 39pf/m  1nF na 20mm 100nF
    """

    training_shell. model = PRE.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_pres.pkl'), DEVICE, scale=0.3)
    training_shell.batch_size = 128
    training_shell.hidden_size = 384 # defaults to 768
    training_shell.intermediate_size = 1536 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 4 # defaults to 12
    training_shell.num_attention_heads = 4 # defaults to 12
    training_shell.dropuot = 0.1 # default is 0.1
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='pred3', force=False )
    training_shell.train()    

def main():
    trainer()

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()