import torch
torch.autograd.set_detect_anomaly(False)

from szx81.models.bert_rv.dataset import Data
 
# from szx81.data.forecast import Forecast
from szx81.models.bert_rv.cls.train import Train
from szx81.models.bert_rv.cls.model import MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

bert = Train(
    file_name='bert_rv',
    tokenizer_file=r'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;',  
    prediction_file=r'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;',
    model=MODEL,
    data_slice=(0, 100000), # (0.5, None),
    batch_size=32,
    d_model=16,
    intermediate_size=128,
    dropout=0.5,
)

def main():
    bert.train(DEVICE, name_prep=None)

'''DEFAULTS:
    d_model=64,
    intermediate_size=256,
    dropout=0.3,
    data_slice=(None, None), # (0.5, None)
    weight=None, 
    data_split={'train_val': 0.5, 'val_test': 0.75},
    resampler=None,
    batch_size=16,
    model_config_or_path=None,
    criterion=CrossEntropyLoss,
'''

# python -m szx81.models.bert_rv.cls.trainings
if __name__ == "__main__":
     main()