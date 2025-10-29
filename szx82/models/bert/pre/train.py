import os
from os import path
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import szx81.core.core_ as co
from szx81.data_segments.data_segment import DataSegments
from szx81.models.bert_rv.dataset import Data
from szx81.models.bert_rv.dataset import Dataset
from szx81.models.bert_rv.pre.model import Config
from szx81.models.model_shell import ModelShell
from szx81.models.project_shell import ProjectShell

class Train:
    def __init__(
            self,
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
            ):
        self.prediction_file = prediction_file
        self.tokenizer_file = tokenizer_file
        self.data_slice = data_slice
        
        self.model = model
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.pretrained_path = pretrained_path
        self.file_name = file_name        

        self.weight = weight
        self.data_split = data_split
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = None
        self.lr = lr

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
        print(f'''
train data size: {len(data_object.train_data)}
validation data size: {len(data_object.val_data)}
test data size: {len(data_object.test_data)}
vocab hash: {data_object.vocab_hash}''')
    
        self.config = Config(
            vocab_size=data_object.tokenizer.vocab_size(),
            hidden_size=self.hidden_size, # Dimensionality of the encoder
            # layers and the pooler layer.

            num_hidden_layers=self.num_hidden_layers, # Number of hidden
            # layers in the Transformer encoder.
            # Each hidden layer consists of:
            #   Multi-head self-attention mechanism: Captures relationships 
            #       between tokens.
            #   Feed-forward neural network: Processes the output of the 
            #       attention mechanism.
            #   Layer normalization and residual connections: Ensure stability 
            #       and better gradient flow.
            # 
            # Each hidden layer outputs a hidden state for every token in the 
            # input sequence. These hidden states are tensors of shape:
            # (batch_size, sequence_length, config.hidden_size)

            num_attention_heads=self.num_attention_heads,
            # Multiple heads divide embedding dimension (hidden_size) across 
            # them. 

            intermediate_size=self.intermediate_size, #  Dimensionality of 
            # the "intermediate" layer in the encoder.
            hidden_act="gelu", # The non-linear activation function 
            # (function or string) in the encoder and pooler.
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            max_position_embeddings=data_object.seq_len,
            type_vocab_size=2, # The vocabulary size of the `token_type_ids`
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,

            # hidden_dim=self.hidden_size, # nie ma tego. hidden_size
            # num_labels=data_object.num_labels,
            # problem_type='single_label_classification',

            device=self.device,
            crit_weight=self.weight,
            criterion=self.criterion,
            # fix train-time vocab:                   
            vocab=data_object.vocab_hash, 
            pretrained_path = self.pretrained_path
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
                config=self.config,
                lr=self.lr,        
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