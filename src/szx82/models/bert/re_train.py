from os import path
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from szx82.models.model_shell import ModelShell
from szx82.models.project_shell import ProjectShell

class Dataset(Dataset):
    def __init__(self, data,):
        super().__init__()
        self.data = data
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]
    
def data(data_or_file, device='cpu', dtype=torch.long, scale=1):
    if isinstance(data_or_file, str):
        with open(data_or_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = data_or_file
    
    def process(data_set):
        data_set_ = []
        data_end = int(len(data_set) * scale)
        for _ in data_set[:data_end]:

            input_ = {}
            for k, v in _['input'].items():
                input_[k] = torch.tensor(v, dtype=dtype, device=device)
            
            data_set_.append({'input': input_, 'admin': _['admin']})
        return data_set_
    
    data['train_data'] = process(data['train_data'])
    data['val_data'] = process(data['val_data'])
    data['test_data'] = process(data['test_data'])
      
    return data

class ReTrain:
    def __init__(
            self,
            data_store,
            data,
            model,      
            batch_size=256,
            pretrained_path=None,
            config_diff={},
            device = None,
            lr=0.0004,
            stop_thd=0.1,
            ): 
        
        # self.args = locals()

        self.data_store = data_store
        self.model = model
        self.pretrained_path = pretrained_path
        self.config_diff = config_diff
        self.data = data        
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.stop_thd = stop_thd
        self.data_param = None

    def args_dict(self):
        return {
            'data_store': self.data_store,
            'model': str(self.model),
            'pretrained_path': self.pretrained_path,
            'data': self.data_param,
            'batch_size': self.batch_size,
            'lr': self.lr,
        }

    def config_or_path(self):
        self.config_diff.update(
            {
            'num_labels': self.data_param['num_labels'],
            'device': self.device,
            })
        return (self.pretrained_path, self.config_diff)

    def set_project_shell(self, device, name_prep=None):
        self.device = device
        
        train_dataset = Dataset(self.data['train_data'])
        assert len(train_dataset) > 0
        val_dataset = Dataset(self.data['val_data'])
        assert len(val_dataset) > 0
        self.data_param = self.data['parameters']

        print(f'''
train data size: {len(train_dataset)}
validation data size: {len(val_dataset)}
test data size: {len(self.data['test_data'])}
train-val overlap: {self.data_param['train_val_overlap']:1.1e}
vocab hash: {self.data_param['vocab_hash']}''')
        
        model_shell = ModelShell(
                train_dataloader=DataLoader(
                                    train_dataset,
                                    batch_size=self.batch_size, 
                                    shuffle=True,  
                                    pin_memory=False,
                                    drop_last=False,
                                ),
                val_dataloader=DataLoader(
                                    val_dataset,
                                    batch_size=128,
                                    shuffle=False, 
                                    pin_memory=False,
                                    drop_last=False,
                                ),
                ModelClass=self.model,
                config_or_path=self.config_or_path(),
                lr=self.lr,
        )
            
        self.project_shell = ProjectShell(
                model_shell=model_shell,
                store_dir=path.join(
                    self.data_store, self.data_param['out_file_name']),
                name_prep=name_prep,
                stop_thd=self.stop_thd
            )

    def init(self, device, name_prep=None, force=False):
        self.set_project_shell(device, name_prep)

        config = self.project_shell.ms.model_env.model.config
        mpe = config.max_position_embeddings
        msg = f'''
model.config.max_position_embeddings: {mpe}
self.data.seq_len: {self.data['parameters']['seq_len']}
'''
        assert mpe == self.data['parameters']['seq_len'], msg

        self.project_shell.file_exists(force=force)
        self.project_shell.save_project(self.args_dict())
    
    def train(self):
        self.project_shell.train()

def main():
    ReTrain()

# python -m szx82.transformer.bert.train
if __name__ == "__main__":
     main()

# sum(current_cumulated['nsp_eq']) / sum(current_cumulated['nsp_count']) - 0.5

'''
{
'data_store':       
    'C:\\Users\\stefa\\Documents\\workspaces\\szx81\\EURUSD\\data_store', 
'model': "<class 'szx82.models.bert.pre.model.MODEL'>", 
'pretrained_path': None, 
'data': {
    'vocab_size': 41, 
    'vocab_hash': 'ec05018401456653', 
    'seq_len': 103, 
    'num_labels': 8, 
    'virtual_levels': [...], 
    'out_file_name': '(0,),aligned-l,straight_piecewise;mean_len-15;seg_size-10;idx_step-1;', 
    'forecast_file': 'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;', 
    'data_style': {...}
    }, 
    'batch_size': 128, 
    'lr': 0.0004
}

TypeError: keys must be str, int, float, bool or None, not tuple
'''