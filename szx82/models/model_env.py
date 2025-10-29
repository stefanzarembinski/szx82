from torch import nn

class MODEL(nn.Module):
    FILE_PREFIX = 'Out'
    def __init__(self,
                Model,
                config_or_model,
                shell=None,                 
                *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.shell = shell
        if isinstance(config_or_model, Model):
            self.model = config_or_model
            self.config = self.model.config
            assert self.shell.transformer_shell.data_object.vocab_hash \
                == self.config.vocab, \
                    'Train-time vocab differs from the current one!'
        else:
            self.config = config_or_model
            self.model = Model(self.config).to(self.config.device)
        self.device = self.config.device 
        self.set_criterion(self.config.crit_weight)    

    def forward(self, batch):
        loss = None # = model_out.loss
        return loss
    
    def set_criterion(self, weight):
        raise NotImplemented()
    
    def set_weight(self, weight):
        if not weight:
            return
        if not hasattr(self.config, 'crit_weight'):
            return
        try:
            if self.config.crit_weight:
                weight =  self.config.crit_weight * weight
            
            self.set_criterion(weight)

        except Exception as ex:
            print('Cannot change criterion weight:\n' + str(ex))
            return None 
        
    def cumulate(self, current):
        cc = {}
        for k in list(current[0].keys()):
            cc[k] = []
            for _ in current:
                cc[k].append(_[k])
        return cc
    
    def set_results(self, train, val, test=None):
        self.train_results = train
        self.val_results = val
        if test is not None:
            self.test_results = test
    
    def final_adj(self):
        pass

    def accuracy(self, current_cumulated=None):
        return 0
    