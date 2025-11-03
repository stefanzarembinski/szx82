import os
from os import path
import pickle
from torch.utils.data import Dataset, DataLoader

from szx82.models.bert.pre.model import Config
from szx82.models.model_shell import ModelShell
from szx82.models.project_shell import ProjectShell

class Dataset(Dataset):
    def __init__(
                self,
                data,
    ):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # print(self.data[index])
        return self.data[index]
    
def data(data_file):
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    return data

class Train:
    def __init__(
            self,
            file_name,
            data_store,
            data,
            model,         
            hidden_size=128,
            intermediate_size=4 * 128, # 4 * hidden_size?
            num_hidden_layers=6,
            num_attention_heads=2,
            dropout=0.5,
            weight=None, 
            data_split={'train_val': 0.75, 'val_test': 0.25},
            batch_size=16,
            pretrained_path=None,
            criterion=None,
            lr=0.0004,
            ):
        self.file_name = file_name
        self.data_store = data_store
        
        self.model = model
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.pretrained_path = pretrained_path
        self.data = data        

        self.weight = weight
        self.data_split = data_split
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = None
        self.lr = lr

    def set_project_shell(self, device, name_prep=None):
        self.device = device

        if name_prep is not None:
            self.file_name += '_' + name_prep
        
        train_dataset = Dataset(self.data['train_data'])
        val_dataset = Dataset(self.data['val_data'])
        parameters = self.data['parameters']

        print(f'''
train data size: {len(train_dataset)}
validation data size: {len(val_dataset)}
test data size: {len(self.data['test_data'])}
vocab hash: {parameters['vocab_hash']}''')
    
        self.config = Config(
            vocab_size=parameters['vocab_size'],
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
            max_position_embeddings=parameters['seq_len'],
            type_vocab_size=2, # The vocabulary size of the `token_type_ids`
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,

            # hidden_dim=self.hidden_size, # nie ma tego. hidden_size
            # num_labels=parameters.num_labels,
            # problem_type='single_label_classification',

            device=self.device,
            crit_weight=self.weight,
            criterion=self.criterion,
            # fix train-time vocab:                   
            vocab=parameters['vocab_hash'], 
            pretrained_path = self.pretrained_path
        )

        model_shell = ModelShell(
                train_dataloader=DataLoader(
                                    train_dataset,
                                    batch_size=len(train_dataset) \
                                        if self.batch_size is None \
                                            else self.batch_size, 
                                    shuffle=True, 
                                    pin_memory=False,
                                    drop_last=True,
                                ),
                val_dataloader=DataLoader(
                                    val_dataset,
                                    batch_size=len(val_dataset) \
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
            
        self.project_shell = ProjectShell(
                model_shell=model_shell,
                store_dir=path.join(
                    self.data_store, parameters['data_file_name']),
                file_name=self.file_name,
            )

    def init(self, device, name_prep=None):
        self.set_project_shell(device, name_prep)
        self.project_shell.file_exists()
    
    def train(self, print_menu=True): 
        self.project_shell.train(print_menu=print_menu)

def main():
    Train()

# python -m szx82.transformer.bert.train
if __name__ == "__main__":
     main()