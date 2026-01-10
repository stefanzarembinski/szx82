import os.path as path
from os import path
import torch

print(f"`torch` default threads: {torch.get_num_threads()}")
torch.set_num_threads(8) # Replace 8 with the number of physical cores
torch.set_num_interop_threads(8)
print(f"`torch` updated threads: {torch.get_num_threads()}")

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

#     training_shell. model = PRE.MODEL
#     prep='half'

#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_4-8-8-8-8_tokenizer_piecewise;mean_len-15;seg_size-10;idx_step-1;style-narrow;_(0,).pkl'
#             ), DEVICE, scale=1)
#     training_shell.batch_size = 256
#     training_shell.hidden_size = 384 # defaults to 768
#     training_shell.intermediate_size = 1536 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 6 # defaults to 12
#     training_shell.num_attention_heads = 6 # defaults to 12
#     training_shell.dropuot = 0.3 # default is 0.1
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train() 

# # training; loss trn,val:0.29,0.28; best:1.40; train:nsp,mlm:0.70,0.68; val.:nsp,mlm:0.70,0.71; ep:6; batch:252/938;
# # training; loss trn,val:0.29,0.28; best:1.41; train:nsp,mlm:0.71,0.69; val.:nsp,mlm:0.70,0.71; ep:7; batch:56/938;
# # training; loss trn,val:0.28,0.28; best:1.43; train:nsp,mlm:0.73,0.70; val.:nsp,mlm:0.72,0.71; ep:8; batch:28/938;
# # training; loss trn,val:0.27,0.28; best:1.45; train:nsp,mlm:0.75,0.70; val.:nsp,mlm:0.73,0.72; ep:9; batch:43/938;
# # training; loss trn,val:0.26,0.27; best:1.46; train:nsp,mlm:0.77,0.70; val.:nsp,mlm:0.74,0.72; ep:10; batch:1/938;
# # training; loss trn,val:0.25,0.27; best:1.47; train:nsp,mlm:0.78,0.71; val.:nsp,mlm:0.75,0.72; ep:11; batch:28/938;
# # training; loss trn,val:0.25,0.27; best:1.48; train:nsp,mlm:0.80,0.71; val.:nsp,mlm:0.76,0.72; ep:12; batch:203/938;
# # training; loss trn,val:0.22,0.29; best:1.51; train:nsp,mlm:0.87,0.73; val.:nsp,mlm:0.76,0.73; ep:16; batch:833/938;
# # training; loss trn,val:0.21,0.29; best:1.52; train:nsp,mlm:0.88,0.73; val.:nsp,mlm:0.78,0.73; ep:17; batch:680/938;
# # training; loss trn,val:0.20,0.29;  :1.52; train:nsp,mlm:0.89,0.73; val.:nsp,mlm:0.79,0.74; ep:18; batch:25/938;


#     training_shell. model = PRE.MODEL
#     prep='third'

#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_4-8-8-8-8_tokenizer_piecewise;mean_len-15;seg_size-10;idx_step-1;style-narrow;_(0,).pkl'
#             ), DEVICE, scale=1)
#     training_shell.batch_size = 256
#     training_shell.hidden_size = 256 # defaults to 768
#     training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 4 # defaults to 12
#     training_shell.num_attention_heads = 4 # defaults to 12
#     training_shell.dropuot = 0.3 # default is 0.1
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train() 
# # train data size: 57848
# # validation data size: 14476
# # test data size: 72296

# # training; loss trn,val:0.15,0.34; best:1.57; train:nsp,mlm:0.97,0.75; val.:nsp,mlm:0.82,0.73; ep:111; batch:133/226;
# # training; loss trn,val:0.15,0.35; best:1.57; train:nsp,mlm:0.97,0.76; val.:nsp,mlm:0.82,0.74; ep:123; batch:140/226;

#     training_shell. model = PRE.MODEL
#     prep='third1'

#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_4-8-8-8-8_tokenizer_piecewise;mean_len-15;seg_size-10;idx_step-1;style-narrow;_(0,).pkl'
#             ), DEVICE, scale=1)
#     training_shell.batch_size = 128 #256
#     training_shell.hidden_size = 256 # defaults to 768
#     training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 4 # defaults to 12
#     training_shell.num_attention_heads = 4 # defaults to 12
#     training_shell.dropuot = 0.3 # default is 0.1
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train() 

# # train data size: 57848
# # validation data size: 14476
# # test data size: 72296

# # training; loss trn,val:0.19,0.31; best:1.55; train:nsp,mlm:0.94,0.72; val.:nsp,mlm:0.83,0.72; ep:59; batch:143/452;
# # validation; loss trn,val:0.15,0.35; best:1.57; train:nsp,mlm:0.97,0.76; val.:nsp,mlm:0.82,0.74; ep:109; batch:113/114;

#     training_shell. model = PRE.MODEL
#     prep='mod'

#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             'data_piecewise;mean_len-15;seg_size-10;idx_step-1;_(0,).pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 128 #256
#     training_shell.hidden_size = 256 # defaults to 768
#     training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 4 # defaults to 12
#     training_shell.num_attention_heads = 4 # defaults to 12
#     training_shell.dropuot = 0.3 # default is 0.1
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 60032
# # validation data size: 15008
# # test data size: 75040

# # training; loss trn,val:0.26,0.37; best:1.28; train:nsp,mlm:0.76,0.71; val.:nsp,mlm:0.54,0.72; ep:97; batch:236/469;

#     training_shell. model = PRE.MODEL
#     prep='mod'

#     training_shell.data = data(
#         path.join(
#             DATA_STORE,
#             '(0,),aligned-r_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), DEVICE, scale=0.5)
#     training_shell.batch_size = 128 #256
#     training_shell.hidden_size = 256 # defaults to 768
#     training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
#     training_shell.num_hidden_layers = 4 # defaults to 12
#     training_shell.num_attention_heads = 4 # defaults to 12
#     training_shell.dropuot = 0.3 # default is 0.1
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep=prep, force=False )
#     training_shell.train()

# # train data size: 60004
# # validation data size: 15008
# # test data size: 75012

# #  training; loss trn,val:0.22,0.32; best:1.46; train:nsp,mlm:0.89,0.71; val.:nsp,mlm:0.75,0.71; ep:81; batch:82/469;
# # training; loss trn,val:0.21,0.34; best:1.47; train:nsp,mlm:0.90,0.71; val.:nsp,mlm:0.75,0.71; ep:89; batch:103/469;
# # training; loss trn,val:0.21,0.34; best:1.47; train:nsp,mlm:0.90,0.71; val.:nsp,mlm:0.75,0.72; ep:92; batch:45/469;

    training_shell. model = PRE.MODEL
    prep='mod'

    training_shell.data = data(
        path.join(
            DATA_STORE,
            '(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
            ), DEVICE, scale=0.5)
    training_shell.batch_size = 128 #256
    training_shell.hidden_size = 256 # defaults to 768
    training_shell.intermediate_size = 1024 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 4 # defaults to 12
    training_shell.num_attention_heads = 4 # defaults to 12
    training_shell.dropuot = 0.3 # default is 0.1
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep=prep, force=False )
    training_shell.train()

 
def main():  
    trainer() 

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()