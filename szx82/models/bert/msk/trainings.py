import torch
torch.autograd.set_detect_anomaly(False)

from szx81.models.bert_rv.dataset import Data
 
# from szx81.data.forecast import Forecast
from szx81.models.bert_rv.msk.train import Train
from szx81.models.bert_rv.msk.model import MODEL

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

bert_rv = Train(
    file_name='bert_rv',
    tokenizer_file=r'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;',  
    prediction_file=r'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;',
    model=MODEL,
    data_slice=(0, 100000), # (0.5, None),
    batch_size=32,
)
# python -m szx81.models.bert_rv.trainings
def main():
    bert_rv.train(DEVICE, name_prep=None)

'''DEFAULTS:
 
'''

#  python -m szx81.models.bert_rv.trainings
if __name__ == "__main__":
     main()