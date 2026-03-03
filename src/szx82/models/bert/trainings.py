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
from szx81.models.bert.indexer.mlm_nsp import MlmNspDto, MlmNsp

DATA_STORE = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store'
DATA_STORE_G = 'G:/My Drive/workplaces/szx81/EURUSD/data_store'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainer():
    training_shell = TRAIN(
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

#     training_shell. model = PRE.MODEL
#     prep='DOWN' 
 
#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 2 # defaults to 12
#     training_shell.num_attention_heads = 2 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.02 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 80000
# # validation data size: 20000
# # test data size: 0
# # vocab hash: ec05018401456653

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_DOWN
# # project name: BERT_PRE_DOWN

# #  training; loss trn,val:0.42,0.40; best:4:1.18; train:nsp,mlm:0.61,0.52; val.:nsp,mlm:0.63,0.55; ep:5; batch:43/313;
# # validation; loss trn,val:0.35,0.33; best:36:1.29; train:nsp,mlm:0.63,0.63; val.:nsp,mlm:0.63,0.66; ep:45; batch:156/157;       Training Time: 14032.25 s

#     training_shell. model = PRE.MODEL
#     prep='UP' 
 
#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 2 # defaults to 12
#     training_shell.num_attention_heads = 2 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.02 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 80000
# # validation data size: 20000
# # test data size: 0
# # vocab hash: ec05018401456653

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_DOWN
# # project name: BERT_PRE_DOWN

# #  validation; loss trn,val:0.35,0.33; best:36:1.29; train:nsp,mlm:0.63,0.63; val.:nsp,mlm:0.63,0.66; ep:45; batch:156/157;       Training Time: 14032.25 s

# # C:\Users\stefa>python -m szx82.models.bert.trainings

# # train data size: 80000
# # validation data size: 20000
# # test data size: 0
# # vocab hash: ec05018401456653

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_UP
# # project name: BERT_PRE_UP

# #  validation; loss trn,val:0.32,0.31; best:122:1.33; train:nsp,mlm:0.68,0.65; val.:nsp,mlm:0.65,0.68; ep:152; batch:156/157;       Training Time: 46920.07 s

#     training_shell. model = PRE.MODEL
#     prep='UP' 
 
#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\WORKSP~1\szx81\EURUSD\DATA_S~1\MLMNSP~1\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 2 # defaults to 12
#     training_shell.num_attention_heads = 2 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.02 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 80000
# # validation data size: 20000
# # test data size: 0
# # vocab hash: ec05018401456653

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_UP
# # project name: BERT_PRE_UP

# # validation; loss trn,val:0.35,0.33; best:38:1.29; train:nsp,mlm:0.64,0.62; val.:nsp,mlm:0.64,0.65; ep:47; batch:156/157;       Training Time: 14453.30 s


    training_shell. model = PRE.MODEL
    prep='DOWN' 
 
    training_shell.data = data(
        path.join(
            r'C:\Users\stefa\Documents\WORKSP~1\szx81\EURUSD\DATA_S~1\MLMNSP~3\MLMNSP~1.PKL'
            ), DEVICE, scale=0.5)
    training_shell.batch_size = 256 #256
    training_shell.hidden_size = 192 # defaults to 768
    training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 2 # defaults to 12
    training_shell.num_attention_heads = 2 # defaults to 12
    training_shell.dropuot = 0.5 # default is 0.1
    training_shell.stop_thd=0.02 # val_loss = 0.34; rain_loss = 0.32
    training_shell.init(DEVICE, name_prep=prep, force=False )
    training_shell.train()

# train data size: 80000
# validation data size: 20000
# test data size: 0
# vocab hash: ec05018401456653

# project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_DOWN
# project name: BERT_PRE_DOWN

#  validation; loss trn,val:0.35,0.34; best:33:1.27; train:nsp,mlm:0.60,0.63; val.:nsp,mlm:0.61,0.66; ep:41; batch:156/157;       Training Time: 12710.63 s

def main():    
    trainer()    

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()   