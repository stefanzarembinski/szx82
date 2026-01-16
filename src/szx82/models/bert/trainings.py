import os.path as path
from os import path
import torch

# print(f"`torch` default threads: {torch.get_num_threads()}")
# torch.set_num_threads(8) # Replace 8 with the number of physical cores
# torch.set_num_interop_threads(8)
# print(f"`torch` updated threads: {torch.get_num_threads()}")

from szx82.models.bert.train import Train as TRAIN
import szx82.models.bert.pre.model as PRE
from szx82.models.bert.re_train import data, ReTrain as RE_TRAIN
import szx82.models.bert.cls.model as CLS

DATA_STORE = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store'
DATA_STORE_G = 'G:/My Drive/workplaces/szx81/EURUSD/data_store'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainer():
    training_shell = TRAIN(
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

    training_shell. model = PRE.MODEL
    prep='mod-05'

    training_shell.data = data(
        path.join(
            DATA_STORE,
            'C:\\Users\\stefa\\Documents\\workspaces\\szx81\\EURUSD\\data_store\\(0,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;\\(0,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
            ), DEVICE, scale=0.5)
    training_shell.batch_size = 256 #256
    training_shell.hidden_size = 256 # defaults to 768
    training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 4 # defaults to 12
    training_shell.num_attention_heads = 4 # defaults to 12
    training_shell.dropuot = 0.5 # default is 0.1
    training_shell.stop_thd=0.1 
    training_shell.init(DEVICE, name_prep=prep, force=False )
    training_shell.train()
 
# train data size: 252000
# validation data size: 63000 
# test data size: 105000
# train-val overlap: 6.6e-03

# training; loss trn,val:0.42,0.34; best:1.35; train:nsp,mlm:0.69,0.50; val.:nsp,mlm:0.76,0.59; ep:2; batch:500/985; 
# training; loss trn,val:0.34,0.29; best:1.46; train:nsp,mlm:0.76,0.59; val.:nsp,mlm:0.83,0.63; ep:3; batch:368/985;
# training; loss trn,val:0.30,0.26; best:1.51; train:nsp,mlm:0.81,0.61; val.:nsp,mlm:0.86,0.65; ep:4; batch:65/985; 
# training; loss trn,val:0.28,0.23; best:1.58; train:nsp,mlm:0.85,0.63; val.:nsp,mlm:0.91,0.66; ep:5; batch:470/985;
# validation; loss trn,val:0.24,0.22; best:1.60; train:nsp,mlm:0.91,0.65; val.:nsp,mlm:0.93,0.68; ep:6; batch:396/493;
# training; loss trn,val:0.24,0.20; best:1.63; train:nsp,mlm:0.91,0.65; val.:nsp,mlm:0.94,0.69; ep:7; batch:2/985;
# training; loss trn,val:0.23,0.20; best:1.64; train:nsp,mlm:0.92,0.66; val.:nsp,mlm:0.95,0.69; ep:8; batch:461/985;
# training; loss trn,val:0.22,0.18; best:1.67; train:nsp,mlm:0.93,0.66; val.:nsp,mlm:0.97,0.70; ep:9; batch:170/985;
# training; loss trn,val:0.21,0.18; best:1.67; train:nsp,mlm:0.94,0.67; val.:nsp,mlm:0.96,0.71; ep:10; batch:248/985;
# training; loss trn,val:0.21,0.18; best:1.67; train:nsp,mlm:0.95,0.68; val.:nsp,mlm:0.96,0.71; ep:11; batch:604/985;
# training; loss trn,val:0.20,0.17; best:1.69; train:nsp,mlm:0.95,0.68; val. :nsp,mlm:0.97,0.72; ep:12; batch:5/985;
# training; loss trn,val:0.20,0.17; best:1.69; train:nsp,mlm:0.96,0.68; val.:nsp,mlm:0.97,0.72; ep:13; batch:190/985;  
#  training; loss trn,val:0.19,0.16; best:1.70; train:nsp,mlm:0.96,0.69; val.:nsp,mlm:0.98,0.73; ep:14; batch:3/985;
# training; loss trn,val:0.19,0.17; best:1.70; train:nsp,mlm:0.96,0.69; val.:nsp,mlm:0.97,0.73; ep:15; batch:159/985;
# training; loss trn,val:0.16,0.14; best:1.75; train:nsp,mlm:0.98,0.73; val.:nsp,mlm:0.99,0.77; ep:25; batch:101/985;
# training; loss trn,val:0.16,0.14; best:1.76; train:nsp,mlm:0.98,0.73; val.:nsp,mlm:0.98,0.77; ep:27; batch:861/985;

 
def main():   
    trainer()  

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()  