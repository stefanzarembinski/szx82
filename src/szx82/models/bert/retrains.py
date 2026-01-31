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


#     training_shell.model = PRE.MODEL
#     prep_orig = ''
#     prep='26cont'

#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
#             ), 
#         DEVICE, 
#         scale=1
#         )
#     training_shell.pretrained_path = path.join(
#         r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\BERT_PRE_26-bis-bis',
#         'model_bst_.pt'
# )
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.02
#     training_shell.init(DEVICE, name_prep=prep, force=False)
#     training_shell.train()

# # train data size: 120000
# # validation data size: 30000
# # test data size: 50000
# # training; loss trn,val:0.94,0.94; best:42:1.37; train:nsp,mlm:0.69,0.65; val.:nsp,mlm:0.69,0.68; ep:43; batch:844/938; 


    training_shell.model = CLS.MODEL
    prep_orig = ''
    prep='26'

    training_shell.data = data(
        path.join(
            r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS__(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\CLS__(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;.pkl'
            ), 
        DEVICE, 
        scale=1
        )
    training_shell.pretrained_path = path.join(
        r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\sym1_(0,),aligned-l_0-300_piecewise;mean_len-15;seg_size-10;idx_step-1;\BERT_PRE_26cont',
        'model_bst_.pt'
)
    training_shell.batch_size = 128
    training_shell.stop_thd=0.02
    training_shell.init(DEVICE, name_prep=prep, force=False)
    training_shell.train()



def main():
    trainer() 

# python -m szx82.models.bert.retrains
if __name__ == '__main__':
    main() 