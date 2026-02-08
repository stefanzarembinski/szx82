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
from szx81.models.bert.data.mlm_nsp import MlmNspDto, MlmNsp

DATA_STORE = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store'
DATA_STORE_G = 'G:/My Drive/workplaces/szx81/EURUSD/data_store'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def trainer():
    training_shell = TRAIN(
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

    # training_shell. model = PRE.MODEL
    # prep='mod-05'

    # training_shell.data = data(
    #     path.join(
    #         DATA_STORE,
    #         'C:\\Users\\stefa\\Documents\\workspaces\\szx81\\EURUSD\\data_store\\(0,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;\\(0,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
    #         ), DEVICE, scale=0.5)
    # training_shell.batch_size = 256 #256
    # training_shell.hidden_size = 256 # defaults to 768
    # training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
    # training_shell.num_hidden_layers = 4 # defaults to 12
    # training_shell.num_attention_heads = 4 # defaults to 12
    # training_shell.dropuot = 0.5 # default is 0.1
    # training_shell.stop_thd=0.1 
    # training_shell.init(DEVICE, name_prep=prep, force=False )
    # training_shell.train()
 
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
# training; loss trn,val:0.16,0.14; best:1.76; train:nsp,mlm:0.98,0.73; val.:nsp,mlm:0.98,0.77; ep:2 7; batch:861/985;
# training; loss trn,val:0.16,0.14; best:1.76; train:nsp,mlm:0.98,0.74; val.:nsp,mlm:0.98,0.78; ep:29; batch:127/985;
# training; loss trn,val:0.15,0.13; best:1.77; train:nsp,mlm:0.98,0.74; val.:nsp,mlm:0.98,0.78; ep:30; batch:35/985;
# training; loss trn,val:0.15,0.14; best:1.77; train:nsp,mlm:0.99,0.74; val.:nsp,mlm:0.98,0.79; ep:31; batch:748/985;
# training; loss trn,val:0.15,0.13; best:1.77; train:nsp,mlm:0.99,0.75; val.:nsp,mlm:0.98,0.79; ep:32; batch:810/985;
# training; loss trn,val:0.15,0.13; best:1.78; train:nsp,mlm:0.99,0.75; val.:nsp,mlm:0.99,0.79; ep:33; batch:909/985;
# training; loss trn,val:0.15,0.12; best:1.78; train:nsp,mlm:0.99,0.75; val.:nsp,mlm:0.99,0.79; ep:34; batch:151/985;
#  training; loss trn,val:0.15,0.13; best:1.78; train:nsp,mlm:0.99,0.75; val.:nsp,mlm:0.98,0.80; ep:35; batch:1/985;
# training; loss trn,val:0.14,0.12; best:1.78; train:nsp,mlm:0.99,0.76; val.:nsp,mlm:0.99,0.80; ep:36; batch:883/985;
# training; loss trn,val:0.14,0.13; best:1.78; train:nsp,mlm:0.99,0.76; val.:nsp,mlm:0.98,0.80; ep:37; batch:15/985;
# training; loss trn,val:0.14,0.12; best:1.79; train:nsp,mlm:0.99,0.76; val.:nsp,mlm:0.99,0.80; ep:38; batch:303/985;
# training; loss trn,val:0.14,0.12; best:1.79; train:nsp,mlm:0.99,0.76; val.:nsp,mlm:0.98,0.81; ep:39; batch:1/985;
# training; loss trn,val:0.13,0.11; best:1.81; train:nsp,mlm:0.99,0.78; val.:nsp,mlm:0.98,0.82; ep:47; batch:831/985;
# training; loss trn,val:0.13,0.11; best:1.81; train:nsp,mlm:0.99,0.78; val.:nsp,mlm:0.99,0.83; ep:49; batch:689/985;
# training; loss trn,val:0.13,0.11; best:1.81; train:nsp,mlm:0.99,0.78; val.:nsp,mlm:0.98,0.83; ep:50; batch:595/985;
#  training; loss trn,val:0.13,0.11; best:1.82; train:nsp,mlm:0.99,0.78; val.:nsp,mlm:0.99,0.83; ep:51; batch:344/985;
# training; loss trn,val:0.12,0.11; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.99,0.83; ep:52; batch:85/985;
# training; loss trn,val:0.12,0.10; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.99,0.83; ep:53; batch:567/985;
# training; loss trn,val:0.12,0.11; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.99,0.83; ep:54; batch:41/985;
# training; loss trn,val:0.12,0.10; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.99,0.84; ep:55; batch:858/985;
# training; loss trn,val:0.12,0.11; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.98,0.84; ep:56; batch:187/985;
# training; loss trn,val:0.12,0.11; best:1.82; train:nsp,mlm:0.99,0.79; val.:nsp,mlm:0.98,0.84; ep:57; batch:499/985;
# training; loss trn,val:0.12,0.11; best:1.82; train:nsp,mlm:0.99,0.80; val.:nsp,mlm:0.98,0.84; ep:58; batch:765/985;
# validation; loss trn,val:0.11,0.10; best:1.84; train:nsp,mlm:0.99,0.81; val.:nsp,mlm:0.99,0.85; ep:65; batch:400/493;
# training; loss trn,val:0.11,0.10; best:1.84; train:nsp,mlm:0.99,0.81; val.:nsp,mlm:0.99,0.85; ep:66; batch:1/985;
# training; loss trn,val:0.11,0.10; best:1.84; train:nsp,mlm:0.99,0.81; val.:nsp,mlm:0.99,0.85; ep:67; batch:943/985;
# training; loss trn,val:0.11,0.10; best:1.84; train:nsp,mlm:0.99,0.81; val.:nsp,mlm:0.98,0.85; ep:68; batch:4/985;
# training; loss trn,val:0.11,0.10; best:1.84; train:nsp,mlm:0.99,0.81; val.:nsp,mlm:0.98,0.85; ep:68; batch:668/985;
  
 
    # training_shell. model = PRE.MODEL
    # prep=None 
 
    # training_shell.data = data(
    #     path.join(
    #         DATA_STORE,
    #         'C:\\Users\\stefa\\Documents\\workspaces\\szx81\\EURUSD\\data_store\\(7,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;\\(7,),aligned-l,straight_0-60_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
    #         ), DEVICE, scale=0.5)
    # training_shell.batch_size = 256 #256
    # training_shell.hidden_size = 256 # defaults to 768
    # training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
    # training_shell.num_hidden_layers = 4 # defaults to 12
    # training_shell.num_attention_heads = 4 # defaults to 12
    # training_shell.dropuot = 0.5 # default is 0.1
    # training_shell.stop_thd=0.1 
    # training_shell.init(DEVICE, name_prep=prep, force=False )
    # training_shell.train()

    #         # train data size: 252000
    #         # validation data size: 63000
    #         # test data size: 105000
    #         # train-val overlap: 4.0e-03

    #         # training; loss trn,val:0.09,0.08; best:1.88; train:nsp,mlm:0.99,0.85; val.:nsp,mlm:0.98,0.90; ep:68; batch:410/985;


#     training_shell. model = PRE.MODEL
#     prep='22' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\(7,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\(7,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 3 # defaults to 12
#     training_shell.num_attention_heads = 3 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.1 
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 30000
# # validation data size: 7500
# # test data size: 12500

# # training; loss trn,val:0.34,0.33; best:32:1.33; train:nsp,mlm:0.70,0.62; val.:nsp,mlm:0.69,0.64; ep:34; batch:11/118;
# # training; loss trn,val:0.33,0.33; best:42:1.34; train:nsp,mlm:0.71,0.63; val.:nsp,mlm:0.69,0.65; ep:45; batch:112/118; 
# # training; loss trn,val:0.32,0.32; best:58:1.36; train:nsp,mlm:0.73,0.64; val.:nsp,mlm:0.70,0.66; ep:59; batch:5/118;
# # training; loss trn,val:0.31,0.32; best:67:1.36; train:nsp,mlm:0.74,0.64; val.:nsp,mlm:0.69,0.66; ep:71; batch:82/118; 


#     training_shell. model = PRE.MODEL
#     prep='sym' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 3 # defaults to 12
#     training_shell.num_attention_heads = 3 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.015 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 30000
# # validation data size: 7500
# # test data size: 12500

# # training; loss trn,val:0.37,0.36; best:15:1.25; train:nsp,mlm:0.64,0.58; val.:nsp,mlm:0.64,0.60; ep:16; batch:86/118;
# # training; loss trn,val:0.34,0.34; best:33:1.29; train:nsp,mlm:0.66,0.63; val.:nsp,mlm:0.63,0.65; ep:40; batch:69/118;
# #  training; loss trn,val:0.34,0.33; best:41:1.30; train:nsp,mlm:0.67,0.63; val.:nsp,mlm:0.64,0.65; ep:46; batch:7/118;
# # training; loss trn,val:0.34,0.33; best:49:1.31; train:nsp,mlm:0.67,0.64; val.:nsp,mlm:0.65,0.65; ep:50; batch:102/118; 
# # training; loss trn,val:0.32,0.33; best:61:1.31; train:nsp,mlm:0.69,0.65; val.:nsp,mlm:0.64,0.66; ep:68; batch:109/118;
# # training; loss trn,val:0.32,0.33; best:61:1.31; train:nsp,mlm:0.70,0.65; val.:nsp,mlm:0.64,0.66; ep:76; batch:61/118;


#     training_shell. model = PRE.MODEL
#     prep='23' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 3 # defaults to 12
#     training_shell.num_attention_heads = 3 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.015 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 30000
# # validation data size: 7500
# # test data size: 12500

# # training; loss trn,val:0.59,0.46; best:1:1.08; train:nsp,mlm:0.53,0.36; val.:nsp,mlm:0.63,0.45; ep:2; batch:45/118;
# # training; loss trn,val:0.26,0.34; best:181:1.40; train:nsp,mlm:0.86,0.66; val.:nsp,mlm:0.71,0.67; ep:208; batch:47/118;


#     training_shell. model = PRE.MODEL
#     prep='24' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 3 # defaults to 12
#     training_shell.num_attention_heads = 3 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.01 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# #  training; loss trn,val:0.31,0.30; best:48:1.36; train:nsp,mlm:0.69,0.66; val.:nsp,mlm:0.67,0.69; ep:50; batch:83/235;
# # training; loss trn,val:0.31,0.30; best:50:1.37; train:nsp,mlm:0.70,0.66; val.:nsp,mlm:0.67,0.69; ep:58; batch:72/235;
# # training; loss trn,val:0.31,0.30; best:50:1.37; train:nsp,mlm:0.71,0.67; val.:nsp,mlm:0.67,0.69; ep:64; batch:30/235;
# # training; loss trn,val:0.31,0.30; best:50:1.37; train:nsp,mlm:0.71,0.67; val.:nsp,mlm:0.67,0.69; ep:65; batch:165/235;


#     training_shell. model = PRE.MODEL
#     prep='25' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 4 # defaults to 12
#     training_shell.num_attention_heads = 4 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.015 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 60000
# # validation data size: 15000  
# # test data size: 25000

# # training; loss trn,val:0.36,0.34; best:8:1.28; train:nsp,mlm:0.67,0.58; val.:nsp,mlm:0.67,0.61; ep:9; batch:10/235;
# # training; loss trn,val:0.36,0.34; best:9:1.28; train:nsp,mlm:0.66,0.59; val.:nsp,mlm:0.67,0.61; ep:10; batch:66/235;
# # training; loss trn,val:0.35,0.34; best:10:1.29; train:nsp,mlm:0.66,0.60; val.:nsp,mlm:0.66,0.62; ep:11; batch:207/235;
# # training; loss trn,val:0.34,0.32; best:18:1.33; train:nsp,mlm:0.67,0.63; val.:nsp,mlm:0.68,0.65; ep:19; batch:51/235;
# #  validation; loss trn,val:0.33,0.32; best:18:1.33; train:nsp,mlm:0.67,0.63; val.:nsp,mlm:0.68,0.65; ep:19; batch:117/118;


#     training_shell. model = PRE.MODEL
#     prep='26-bis' 
 
#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\sym_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 256 #256
#     training_shell.hidden_size = 192 # defaults to 768
#     training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 2 # defaults to 12
#     training_shell.num_attention_heads = 2 # defaults to 12
#     training_shell.dropuot = 0.5 # default is 0.1
#     training_shell.stop_thd=0.015 # val_loss = 0.34; rain_loss = 0.32
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 60000
# # validation data size: 15000
# # test data size: 25000
# # validation; loss trn,val:0.34,0.32; best:43:1.33; train:nsp,mlm:0.68,0.63; val.:nsp,mlm:0.67,0.66; ep:50; batch:117/118;       Training Time: 12246.74 s

    # training_shell. model = PRE.MODEL
    # prep='26-bis-bis' 
 
    # training_shell.data = data(
    #     path.join(
    #         r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
    #         ), DEVICE, scale=0.5)
    # training_shell.batch_size = 256 #256
    # training_shell.hidden_size = 192 # defaults to 768
    # training_shell.intermediate_size = 768 # 4 * hidden_size? # defaults to 3072
    # training_shell.num_hidden_layers = 2 # defaults to 12
    # training_shell.num_attention_heads = 2 # defaults to 12
    # training_shell.dropuot = 0.5 # default is 0.1
    # training_shell.stop_thd=0.02 # val_loss = 0.34; rain_loss = 0.32
    # training_shell.init(DEVICE, name_prep=prep, force=False )
    # training_shell.train()

#     training_shell. model = PRE.MODEL
#     prep='UP' 
 
#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(7,),aligned-l_0-nan_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\MlmNsp_UP_(7,),aligned-l_0-nan_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl'
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

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(7,),aligned-l_0-nan_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_UP
# # project name: BERT_PRE_UP

# #  validation; loss trn,val:0.31,0.31; best:180:1.33; train:nsp,mlm:0.69,0.66; val.:nsp,mlm:0.63,0.69; ep:200; batch:156/157;       Training Time: 58985.17 s

    training_shell. model = PRE.MODEL
    prep='DOWN' 
 
    training_shell.data = data(
        path.join(
            r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl'
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


def main():    
    trainer()    

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()  