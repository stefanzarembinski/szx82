from os import path
import pickle
import torch
torch.autograd.set_detect_anomaly(False)

from szx82.models.bert.pre.train import Train, data
from szx82.models.bert.pre.model import MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
DATA_STORE = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store'
DATA_FILE = r'data_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;.pkl'

training_shell = Train(
    file_name='bert',
    data_store=DATA_STORE,
    data=None,
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
    # training_shell.data = data(path.join(DATA_STORE, DATA_FILE))
    # # vocab:  RY9oRdHDGk7rFWlsW2RnCDqvJtg=  
    # training_shell.batch_size = 256
    # training_shell.train(DEVICE, name_prep='bs256') 
    # loss trn,val:0.37,0.44 best: 0.62

    # # vocab:  RY9oRdHDGk7rFWlsW2RnCDqvJtg=  
    # training_shell.batch_size = 256
    # training_shell.train(DEVICE, name_prep='ds200k')     
    
    # ##### testing DEVELOPMENT    
    training_shell.data = data(path.join(
        DATA_STORE, 
        r'dev_tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;.pkl'))
    training_shell.batch_size = 64
    training_shell.init(DEVICE, name_prep='dev', force=True)
    training_shell.train()
    
    # ##### testing training CONTINUATION
    # training_shell.model_config_or_path = r'C:\Users\stefa\Documents\workspaces\szx81\EURUSD\data_store\tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;\BERT_PRE_bert_rv\BERT_PRE_bert_rv_bst_.pt'
    # training_shell.train(DEVICE, name_prep='cont')

'''DEFAULTS:
'''

# python -m szx82.models.bert.pre.trainings 
if __name__ == "__main__":
     main() 

"""
BLOG

""" 