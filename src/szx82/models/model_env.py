from torch import nn

class MODEL(nn.Module):
    FILE_PREFIX = 'Out'
    def __init__(self,
                model_factory,
                config_or_model,
                shell=None,                 
                *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.shell = shell
        model = model_factory(config_or_model)
        self.model = model.to(model.config.device)
        self.set_criterion(self.model.config.crit_weight)    

    def forward(self, batch):
        loss = None # = model_out.loss
        return loss
    
    def set_criterion(self, weight):
        raise NotImplemented()
    
    def set_weight(self, weight):
        if not weight:
            return
        if not hasattr(self.model.config, 'crit_weight'):
            return
        try:
            if self.model.config.crit_weight:
                weight =  self.model.config.crit_weight * weight
            
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
    
    def final_adj(self):
        pass

    def accuracy(self, current_cumulated=None):
        return 0
    