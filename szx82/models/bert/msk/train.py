import os
from os import path
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import szx81.core.core_ as co
from szx81.data_segments.data_segment import DataSegments
from szx81.models.bert_rv.dataset import Data
from szx81.models.bert_rv.dataset import Dataset
from szx81.models.bert_rv.msk.model import Config
from szx81.models.model_shell import ModelShell
from szx81.models.project_shell import ProjectShell

class Train:
    def __init__(
            self,
            tokenizer_file,
            file_name,
            model,
            d_model=100,
            feature_count=10,
            hidden_size1=10,
            dropout=0.3,
            prediction_file=r'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;',
            data_slice=(None, None), # (0.5, None)
            feature_rmv=None, # [forex_data.FEATURES.index(forex_data.VOLUME),]
            weight=None, 
            data_split={'train_val': 0.5, 'val_test': 0.75},
            resampler=None,
            batch_size=16,
            model_config_or_path=None,
            criterion=CrossEntropyLoss,
            ):
        self.prediction_file = prediction_file
        self.tokenizer_file = tokenizer_file
        self.data_slice = data_slice
        self.feature_rmv = feature_rmv
        self.resampler=resampler
        
        self.model = model
        self.d_model = d_model
        self.feature_count = feature_count
        self.hidden_size1 = hidden_size1
        self.dropout = dropout
        self.model_config_or_path = model_config_or_path
        self.file_name = file_name        

        self.weight = weight
        self.data_split = data_split
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = None 

    def set_ts(self, device, name_prep=None):
        self.device = device

        if name_prep is not None:
            self.file_name += '_' + name_prep
        store_dir = path.join(co.DATA_STORE, self.tokenizer_file)
        
        try:
            os.mkdir(store_dir)
        except Exception as ex:
            if not isinstance(ex, FileExistsError):
                raise(ex)

        data_object = Data(
                data_file_or_object=self.tokenizer_file, 
                prediction_file=self.prediction_file,
                ModelClass=self.model, 
                data_split=self.data_split,
                data_slice=self.data_slice,
                )
    
        if self.model_config_or_path is None:
            self.model_config_or_path = Config(
                vocab_size=data_object.tokenizer.vocab_size,
                max_position_embeddings=data_object.seq_len,
                hidden_dim=60,
                type_vocab_size=2, # vocabulary size of the token_type_ids
                num_attention_heads=3,
                num_hidden_layers=3,
                dropout=0.3,
                hidden_dropout_prob=0.3,
                intermediate_size=384,
                attention_probs_dropout_prob=0.3,
                device=self.device,
                crit_weight=self.weight,
                criterion=self.criterion,                     
                        )
        
        train_dataset = data_object.train_dataset
        val_dataset = data_object.val_dataset

        model_shell = ModelShell(
                train_dataloader=DataLoader(
                                    train_dataset,
                                    batch_size=len(data_object.train_dataset) \
                                        if self.batch_size is None \
                                            else self.batch_size, 
                                    shuffle=True, 
                                    pin_memory=False,
                                    drop_last=True,
                                ),
                val_dataloader=DataLoader(
                                    val_dataset,
                                    batch_size=len(data_object.val_dataset) \
                                        if self.batch_size is None \
                                            else self.batch_size, 
                                    shuffle=False, 
                                    pin_memory=False,
                                    drop_last=True,
                                ),
                ModelClass=self.model,
                config_or_model_path=self.model_config_or_path,
                device=self.device,               
        )
            
        self.ts = ProjectShell(
                        data_object=data_object,
                        model_shell=model_shell,
                        labels=[],
                        store_dir=store_dir,
                        file_name=self.file_name,
                    )

    def train(self, device, name_prep=None):
        self.set_ts(device, name_prep)
        self.ts.file_exists() 
        self.ts.train()

def main():
    Train()

# python -m szx81.transformer.bert_rv.train
if __name__ == "__main__":
     main()