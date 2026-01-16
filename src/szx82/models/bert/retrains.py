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
    training_shell = RE_TRAIN(
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

#     training_shell.model = CLS.MODEL
#     prep_orig = 'half - Copy'
#     prep='half1'

#     training_shell.data = data(path.join(
#         DATA_STORE,
#             'data_4-8-8_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;_clsf.pkl'
#             ), DEVICE, scale=1)
#     training_shell.pretrained_path = path.join(
#         DATA_STORE, 
#             '4-8-8_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;', 'BERT_PRE_' + prep_orig, 'model_bst_.pt')
#     training_shell.batch_size = 256
#     training_shell.stop_thd=0.3
#     training_shell.init(DEVICE, name_prep=prep, force=False)
#     training_shell.train()
# # training; loss trn,val:1.00,1.00; best:0.16; train:acc:0.15; val.:acc:0.16; ep:1; batch:356/704; 
# # training; loss trn,val:0.93,0.94; best:0.23; train:acc:0.22; val.:acc:0.23; ep:2; batch:91/704;
# # training; loss trn,val:0.93,0.94; best:0.23; train:acc:0.22; val.:acc:0.23; ep:2; batch:687/704;


    training_shell.model = PRE.MODEL
    prep_orig = 'mod'
    prep=''

    training_shell.data = data(
        path.join(
            DATA_STORE,
'(0,),aligned-l,straight_100000_piecewise;mean_len-15;seg_size-10;idx_step-1;',
'(0,),aligned-l,straight_100000_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
            ), 
        DEVICE, 
        scale=1
        )
    training_shell.pretrained_path = path.join(
            DATA_STORE,
'(0,),aligned-l,straight_piecewise;mean_len-15;seg_size-10;idx_step-1;', 
'BERT_PRE_' + prep_orig, 'model_bst_.pt')
    training_shell.batch_size = 128
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep=prep, force=False)
    training_shell.train()

# train data size: 60004
# validation data size: 15008
# test data size: 75012

# training; loss trn,val:1.00,1.00; best:1.15; train:nsp,mlm:0.60,0.53; val.:nsp,mlm:0.59,0.56; ep:1; batch:21/469;
# training; loss trn,val:0.44,0.40; best:1.41; train:nsp,mlm:0.76,0.63; val.:nsp,mlm:0.75,0.67; ep:2; batch:10/469;
# training; loss trn,val:0.34,0.46; best:1.43; train:nsp,mlm:0.91,0.66; val.:nsp,mlm:0.75,0.68; ep:4; batch:378/469;


def main():
    trainer() 

# python -m szx82.models.bert.retrains
if __name__ == '__main__':
    main() 