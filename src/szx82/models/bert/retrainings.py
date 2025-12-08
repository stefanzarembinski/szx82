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
        data_store=DATA_STORE,
        data=None,
        model=None,
    )

    """
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

# training; loss trn,val:0.60,0.63; best:0.95; train:nsp,mlm:0.75,0.20; val.:nsp,mlm:0.75,0.20; ep:6; batch:239/844;
    """

    """
    training_shell.model = CLS.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,
            'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_clsd.pkl'), DEVICE, scale=1)
    training_shell.pretrained_path = path.join(
        DATA_STORE_G, 
        path.join(
            'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;/BERT_PRE_bert_pre', 'BERT_PRE_bert_pre_bst_.pt'))
    training_shell.batch_size = 128
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='clsd', force=False)
    training_shell.train()
# training; loss trn,val:0.46,0.48; best:0.82; train:acc:0.82; val.:acc:0.81; ep:21; batch:196/1407;

    """

    """
    training_shell.model = CLS.MODEL
    training_shell.data = data(path.join(
            DATA_STORE,
            'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;_clsd.pkl'), DEVICE, scale=1)
    training_shell.pretrained_path = path.join(
        DATA_STORE, 
        path.join(
            'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;/BERT_CLS_bert_cls4', 'BERT_CLS_bert_cls4_bst_.pt'))
    training_shell.batch_size = 128
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='cls4', force=False)
    training_shell.train()
# training; loss trn,val:1.00,0.99; best:0.82; train:acc:0.82; val.:acc:0.82; ep:6; batch:749/1407;
    """

#     training_shell.model = CLS.MODEL
#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;_clsd.pkl'), DEVICE, scale=0.5)
#     training_shell.pretrained_path = path.join(
#         DATA_STORE, 
#         path.join(
#             'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;/BERT_PRE_bert_nar1', 'BERT_PRE_bert_nar1_bst_.pt'))
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep='nar1', force=False)
#     training_shell.train()
# # training; loss trn,val:0.64,0.66; best:0.81; train:acc:0.82; val.:acc:0.81; ep:5; batch:675/704;
    

#     training_shell.model = CLS.MODEL
#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_8-16-16_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;_clsf.pkl'), DEVICE, scale=0.5)    

#     training_shell.pretrained_path = path.join(
#         DATA_STORE, 
#         path.join(
#             '8-16-16_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;\BERT_PRE__nar1', 'BERT_PRE__nar1_bst_.pt'))
#     training_shell.batch_size = 128
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep='nar1', force=False)
#     training_shell.train()
# # training; loss trn,val:1.00,1.00; best:0.08; train:acc:0.12; val.:acc:0.08; ep:2; batch:50/704;
# # training; loss trn,val:0.64,0.63; best:0.47; train:acc:0.46; val.:acc:0.47; ep:3; batch:430/704;
# # training; loss trn,val:0.67,0.67; best:0.47; train:acc:0.44; val.:acc:0.43; ep:9; batch:233/704;
# # validation; loss trn,val:0.67,0.67; best:0.47; train:acc:0.44; val.:acc:0.43; ep:15; batch:141/176;

#     training_shell.model = CLS.MODEL
#     training_shell.data = data(path.join(
#             DATA_STORE,
#             'data_8-16-16_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;_clsf.pkl'), DEVICE, scale=0.5)    

#     training_shell.pretrained_path = path.join(
#         DATA_STORE, 
#         path.join(
#             '8-16-16_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;\\BERT_CLS__nar1', 'BERT_CLS__nar1_bst_.pt'))
#     training_shell.batch_size = 64
#     training_shell.stop_thd=0.1
#     training_shell.init(DEVICE, name_prep='nar2', force=False)
#     training_shell.train()
# #  training; loss trn,val:1.00,1.00; best:0.47; train:acc:0.46; val.:acc:0.47; ep:1; batch:19/1407;
# # training; loss trn,val:1.05,1.06; best:0.47; train:acc:0.44; val.:acc:0.43; ep:8; batch:819/1407;

    training_shell.model = PRE.MODEL
    training_shell.data = data(path.join(
        DATA_STORE,
            'data_4-8-8_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;_pre.pkl'), DEVICE, scale=1)
    training_shell.pretrained_path = path.join(
        DATA_STORE, 
            '4-8-8_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;style-narrow;', 'BERT_PRE_pre', 'model_bst_.pt')
    training_shell.batch_size = 128
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep='pre', force=False)
    training_shell.train()

def main():
    trainer()

# python -m szx82.models.bert.retrainings
if __name__ == '__main__':
    main() 