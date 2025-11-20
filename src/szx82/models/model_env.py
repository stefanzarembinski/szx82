from torch import nn
from transformers import BertConfig

class MODEL(nn.Module):
    FILE_PREFIX = 'Out'
    def __init__(self,
                BertModel,
                config_or_model,
                shell=None,
                args=None):

        super().__init__()

        self.shell = shell
        model = self.model_factory(BertModel, config_or_model, args)
        self.model = model.to(model.config.device) 

    def model_factory(self, BertModel, config_or_path, args=None):
        if isinstance(config_or_path, BertConfig):
            return BertModel(config_or_path)

        path, config_diff = config_or_path
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=path)
        for k, v in config_diff.items():
            setattr(config, k, v)
        model = BertModel.from_pretrained(
                pretrained_model_name_or_path=path, config=config)
        return model

    def forward(self, batch):
        loss = None # = model_out.loss
        return loss 
        
    def cumulate(self, current):
        cc = {}
        for k in list(current[0].keys()):
            cc[k] = []
            for _ in current:
                cc[k].append(_[k])
        return cc
    
    def final_adj(self):
        pass

    def accuracy(self, current_cumulated=None):
        return 0
    