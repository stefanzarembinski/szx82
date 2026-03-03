import os.path as path
from os import path
import torch

# print(f"`torch` default threads: {torch.get_num_threads()}")
# torch.set_num_threads(8) # Replace 8 with the number of physical cores
# torch.set_num_interop_threads(8)
# print(f"`torch` updated threads: {torch.get_num_threads()}")

from szx81.models.bert.indexer.binary_cls import BinaryClsDto # for pickling

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
#     prep='DOWN'

#     training_shell.data = data(
#         path.join(
#             r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl'
#             ), 
#         DEVICE, 
#         scale=1
#         )
#     training_shell.pretrained_path = path.join(
#         r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_DOWN',
#         'model_bst_.pt'
# )
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.02
#     training_shell.init(DEVICE, name_prep=prep, force=False)
#     training_shell.train()

# # train data size: 80000
# # validation data size: 20000
# # test data size: 0
# # vocab hash: ec05018401456653

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_UP
# # project name: BERT_PRE_UP

# #  validation; loss trn,val:0.32,0.31; best:122:1.33; train:nsp,mlm:0.68,0.65; val.:nsp,mlm:0.65,0.68; ep:152; batch:156/157;       Training Time: 46920.07 s

# # C:\Users\stefa> python -m szx82.models.bert.retrains

# # train data size: 160000
# # validation data size: 39998
# # test data size: 0
# # vocab hash: ec05018401456653
# # Some weights of BertForSequenceClassification were not initialized from the model checkpoint at C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_DOWN\model_bst_.pt and are newly initialized: ['classifier.bias', 'classifier.weight']
# # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_CLS_DOWN
# # project name: BERT_CLS_DOWN

# #  "Y" for overwriting existing file in dir C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_CLS_DOWN, anything else to abort: Y
# #  validation; loss trn,val:0.79,0.88; best:14:0.68; train:acc:0.71; val.:acc:0.68; ep:19; batch:312/313;       Training Time: 4765.18 s

#     training_shell.model = CLS.MODEL
#     prep='DOWN'

#     training_shell.data = data(
#         path.join(
#             r"C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);.pkl"
#             ), 
#         DEVICE, 
#         scale=1
#         )
#     training_shell.pretrained_path = path.join(
#         r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_UP',
#         'model_bst_.pt'
# )
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.02
#     training_shell.init(DEVICE, name_prep=prep, force=False)
#     training_shell.train()

# # train data size: 160000
# # validation data size: 39998
# # test data size: 0
# # vocab hash: ec05018401456653
# # Some weights of BertForSequenceClassification were not initialized from the model checkpoint at C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_PRE_UP\model_bst_.pt and are newly initialized: ['classifier.bias', 'classifier.weight']
# # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(0, 302879);\BERT_CLS_DOWN
# # project name: BERT_CLS_DOWN

# #  validation; loss trn,val:0.73,0.79; best:17:0.73; train:acc:0.75; val.:acc:0.73; ep:19; batch:312/313;       Training Time: 4847.73 s


#     training_shell.model = CLS.MODEL
#     prep='UP'

#     training_shell.data = data(
#         path.join(
#             r"C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);.pkl"
#             ), 
#         DEVICE, 
#         scale=1
#         )
#     training_shell.pretrained_path = path.join(
#         r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_UP',
#         'model_bst_.pt'
# )
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.02
#     training_shell.init(DEVICE, name_prep=prep, force=False)
#     training_shell.train()

# # train data size: 160000
# # validation data size: 39998
# # test data size: 0
# # vocab hash: ec05018401456653
# # Some weights of BertForSequenceClassification were not initialized from the model checkpoint at C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_UP\model_bst_.pt and are newly initialized: ['classifier.bias', 'classifier.weight']
# # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# # project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_UP_(-1,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_CLS_UP
# # project name: BERT_CLS_UP

# #  validation; loss trn,val:0.83,0.86; best:14:0.70; train:acc:0.71; val.:acc:0.69; ep:19; batch:312/313;       Training Time: 4576.21 s


    training_shell.model = CLS.MODEL
    prep='DOWN'

    training_shell.data = data(
        path.join(
            r"C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);.pkl"
            ), 
        DEVICE, 
        scale=1
        )
    training_shell.pretrained_path = path.join(
        r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_DOWN',
        'model_bst_.pt'
)
    training_shell.batch_size = 128
    training_shell.stop_thd=0.02
    training_shell.init(DEVICE, name_prep=prep, force=False)
    training_shell.train()

# train data size: 160000
# validation data size: 39998
# test data size: 0
# vocab hash: ec05018401456653
# Some weights of BertForSequenceClassification were not initialized from the model checkpoint at C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\MlmNsp_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_PRE_DOWN\model_bst_.pt and are newly initialized: ['classifier.bias', 'classifier.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# project dir: C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\CLS_DOWN_(0,),aligned-l_piecewise;mean_len-15;seg_size-10;idx_step-1;data_range-(297120, 733312);\BERT_CLS_DOWN
# project name: BERT_CLS_DOWN

#  validation; loss trn,val:0.84,0.87; best:18:0.66; train:acc:0.67; val.:acc:0.66; ep:19; batch:312/313;       Training Time: 4565.57 s

def main():
    trainer() 

# python -m szx82.models.bert.retrains
if __name__ == '__main__':
    main() 