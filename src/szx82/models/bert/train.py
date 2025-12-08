from szx82.models.bert.pre.model import Config
from szx82.models.bert.re_train import ReTrain

class Train(ReTrain):
    def __init__(
            self, 
            data_store, 
            data, 
            model,  
            batch_size=256,
            device = None,
            hidden_size=384, # defaults to 768
            intermediate_size=4 * 1536, # 4 * hidden_size? # defaults to 3072
            num_hidden_layers=6, # defaults to 12
            num_attention_heads=6, # defaults to 12            
            dropout=0.3,
            lr=0.0004, 
            stop_thd=0.1,
            ):

        super().__init__(data_store, data, model, batch_size, None, {}, device, lr, stop_thd)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout   
        self.device = None

    def config_or_path(self):   
        config = Config(
            vocab_size=self.data_param['vocab_size'],
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
            max_position_embeddings=self.data_param['seq_len'],
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
            # fix train-time vocab:                   
            vocab=self.data_param['vocab_hash'], 
            pretrained_path = self.pretrained_path
        )
        return config

def main():
    Train()

# python -m szx82.transformer.bert.train
if __name__ == "__main__":
     main()