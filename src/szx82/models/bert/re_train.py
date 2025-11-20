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
        # print(self.data[index])
        return self.data[index]
    
def data(data_or_file, device='cpu', dtype=torch.long):
    if isinstance(data_or_file, str):
        with open(data_or_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = data_or_file
    
    def process(data_set):
        data_set_ = []
        for _ in data_set:

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
            file_name,
            data_store,
            data,
            model,         
            data_split={'train_val': 0.75, 'val_test': 0.25},
            batch_size=16,
            pretrained_path=None,
            config_diff={},
            device = None,
            lr=0.0004,
            stop_thd=0.1,
            ):
        self.file_name = file_name
        self.data_store = data_store
        self.model = model
        self.pretrained_path = pretrained_path
        self.config_diff = config_diff
        self.data = data        
        self.data_split = data_split
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.stop_thd = stop_thd
        self.parameters = None

    def config_or_path(self):
        self.config_diff.update(
            {
            'num_labels': self.parameters['num_labels'],
            'device': self.device,
            })
        return (self.pretrained_path, self.config_diff)

    def set_project_shell(self, device, name_prep=None):
        self.device = device

        if name_prep is not None:
            self.file_name += '_' + name_prep
        
        train_dataset = Dataset(self.data['train_data'])
        val_dataset = Dataset(self.data['val_data'])
        self.parameters = self.data['parameters']

        print(f'''
train data size: {len(train_dataset)}
validation data size: {len(val_dataset)}
test data size: {len(self.data['test_data'])}
vocab hash: {self.parameters['vocab_hash']}''')

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
                    self.data_store, self.parameters['data_file_name']),
                file_name=self.file_name,
                stop_thd=self.stop_thd
            )

    def init(self, device, name_prep=None, force=False):
        self.set_project_shell(device, name_prep)
        self.project_shell.file_exists(force=force)
    
    def train(self):
        self.project_shell.train()

def main():
    ReTrain()

# python -m szx82.transformer.bert.train
if __name__ == "__main__":
     main()