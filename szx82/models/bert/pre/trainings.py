from os import path
import torch
torch.autograd.set_detect_anomaly(False)

from szx81.models.bert_rv.dataset import Data
import pickle
 
# from szx81.data.forecast import Forecast
from szx81.models.bert_rv.pre.train import Train
from szx81.models.bert_rv.pre.model import MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

training_shell = Train(
    file_name='bert_rv',
    tokenizer_file=r'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;',  
    prediction_file=r'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;',
    model=MODEL,
)

def restore_project_training(project_file):
    project_shell = None
    try:
        path.exists(project_file)
        with open(project_file, 'rb') as f: 
            project_shell = pickle.load(f)
    except Exception as ex:
        print(f'ERROR!\n{str(ex)}')
    if project_shell is not None:
        project_shell.train()
 
def main():
    # restore_project_training(
    #      r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;\BERT_PRE_bert_rv_dev\BERT_PRE_bert_rv_dev.pkl')

    # # PROJECT
    # # vocab: C5RmtP+wFVQCcc5DEuaD8HVYEwA=
    # training_shell.train(DEVICE, name_prep='base')
    # loss trn/val: 0.49/0.50 best: 0.54: : 248it
    # loss trn/val: 0.49/0.49 best: 0.54: : 249    
    
    # # PROJECT
    # # vocab: C5RmtP+wFVQCcc5DEuaD8HVYEwA=
    # training_shell.train(DEVICE, name_prep=None)
    # loss trn/val: 0.65/0.65 best: 0.21
    #  val. nsp,mlm:0.00,0.17

    # # PROJECT
    # # vocab: RY9oRdHDGk7rFWlsW2RnCDqvJtg=  
    # training_shell.batch_size = 128
    # training_shell.train(DEVICE, name_prep='bs128') 

    # PROJECT
    # vocab:  RY9oRdHDGk7rFWlsW2RnCDqvJtg=  
    training_shell.batch_size = 256
    training_shell.num_hidden_layers = 3
    training_shell.train(DEVICE, name_prep='bs256') 
    
    # # ##### testing DEVELOPMENT
    # training_shell.batch_size = 64
    # training_shell.train(DEVICE, name_prep='dev')
 
    # # C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;\BERT_PRE_bert_rv_dev\BERT_PRE_bert_rv_dev.pkl
    
    # ##### testing training CONTINUATION
    # training_shell.model_config_or_path = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;\BERT_PRE_bert_rv\BERT_PRE_bert_rv_bst_.pt'
    # training_shell.train(DEVICE, name_prep='cont')

'''DEFAULTS:
            tokenizer_file,
            prediction_file,
            file_name,
            model,         
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            dropout=0.5,
            data_slice=(0, 100000), # (0.5, None)
            weight=None, 
            data_split={'train_val': 0.75, 'val_test': 0.25},
            batch_size=16,
            pretrained_path=None,
            criterion=None,
            lr=0.0004,
'''

# python -m szx81.models.bert_rv.pre.trainings 
if __name__ == "__main__":
     main() 

"""
BLOG

""" 