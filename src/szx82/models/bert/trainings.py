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

    training_shell. model = PRE.MODEL
    prep='half'

    training_shell.data = data(path.join(
            DATA_STORE,
            'data_4-8-8-8-8_tokenizer_piecewise;mean_len-15;seg_size-10;idx_step-1;style-narrow;_(0,).pkl'
            ), DEVICE, scale=1)
    training_shell.batch_size = 256
    training_shell.hidden_size = 384 # defaults to 768
    training_shell.intermediate_size = 1536 # 4 * hidden_size? # defaults to 3072
    training_shell.num_hidden_layers = 6 # defaults to 12
    training_shell.num_attention_heads = 6 # defaults to 12
    training_shell.dropuot = 0.3 # default is 0.1
    training_shell.stop_thd=0.1
    training_shell.init(DEVICE, name_prep=prep, force=False )
    training_shell.train() 

# validation; loss trn,val:0.41,0.42; best:1.21; train:nsp,mlm:0.66,0.56; val.:nsp,mlm:0.66,0.55; ep:2; batch:703/704;
# validation; loss trn,val:0.39,0.39; best:1.25; train:nsp,mlm:0.67,0.58; val.:nsp,mlm:0.66,0.59; ep:3; batch:703/704;
# validation; loss trn,val:0.38,0.38; best:1.28; train:nsp,mlm:0.68,0.60; val.:nsp,mlm:0.67,0.61; ep:6; batch:703/704;
# training; loss trn,val:0.38,0.37; best:1.28; train:nsp,mlm:0.68,0.60; val.:nsp,mlm:0.67,0.61; ep:8; batch:131/1407;
# training; loss trn,val:0.38,0.37; best:1.29; train:nsp,mlm:0.69,0.60; val.:nsp,mlm:0.68,0.61; ep:9; batch:19/1407;
# training; loss trn,val:0.37,0.37; best:1.29; train:nsp,mlm:0.70,0.60; val.:nsp,mlm:0.68,0.61; ep:10; batch:791/1407;
# training; loss trn,val:0.36,0.37; best:1.30; train:nsp,mlm:0.72,0.61; val. :nsp,mlm:0.69,0.62; ep:13; batch:24/1407;

# training; loss trn,val:0.29,0.28; best:1.40; train:nsp,mlm:0.70,0.68; val.:nsp,mlm:0.70,0.71; ep:6; batch:252/938;
# training; loss trn,val:0.29,0.28; best:1.41; train:nsp,mlm:0.71,0.69; val.:nsp,mlm:0.70,0.71; ep:7; batch:56/938;


def main(): 
    trainer()

# python -m szx82.models.bert.trainings
if __name__ == '__main__':
    main()