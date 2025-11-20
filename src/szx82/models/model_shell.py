import numpy as np
import torch
from torch.optim import Adam

class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = lr

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class ModelShell:
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        ModelClass,
        config_or_path,
        lr=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        **kwargs
    ):
        self.label = {'ModelClass': ModelClass}      
        self.model_env = ModelClass(
            config_or_path=config_or_path,
            shell=self,
            kwargs=kwargs
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.project_shell = None
        self.train_history = {
            'train losses': [],
            'train acc': [],
            'val losses': [],
            'val acc': [],
        }
        
        self.optimizer = Adam(
            self.model_env.model.parameters(), 
            lr=lr, betas=betas, 
            weight_decay=weight_decay
        )

        self.optim_schedule = ScheduledOptim(
            self.optimizer, 
            lr=lr, 
            n_warmup_steps=warmup_steps
        )

    def train(self):
        train_loss = None
        train_loss0 = None
        
        val_loss = None
        val_loss0 = None
        
        best = None
        best_thd = 1e-4 
        ncols=80
        train_acc = None
        val_acc = None
        epochs = 0
        first_epoch = True

        self.model_env.project_shell = self.project_shell
        self.model_env.final_adj()

        def print_progress(what, epoche, idx, dataloader):
            if train_loss0 and val_loss0:
                _best = 'none' if best is None else f'{best:0.2f}'
                print(
                    '\r',
                    f'{what};',
                    f'loss trn,val:{train_loss / train_loss0:4.2f},{val_loss / val_loss0:4.2f};',
                    f'best:{_best};',
                    f'train:{train_acc["msg"]};',
                    f'val.:{val_acc["msg"]};',
                    f'ep:{epoche};',
                    f'batch:{idx}/{len(dataloader)};',
                    '      ',
                    end='')
            else:
                print(
                    '\r',
                    f'{what}; wait...',
                    f'ep:{epoche};',
                    f'batch:{idx}/{len(dataloader)};',
                    '      ',
                    end='')                 

        while True:
            epochs += 1
            self.model_env.model.train()
            train_current = []
            val_current = []            

            train_running_loss = 0 

            for idx, batch in enumerate(self.train_dataloader):
                print_progress('training', epochs, idx, self.train_dataloader)

                self.optimizer.zero_grad()
                loss = self.model_env(batch)
                train_current.append(self.model_env.current)
                train_running_loss += loss.item()

                if not first_epoch:
                    loss.mean().backward()
                    self.optimizer.step()                  
            
            train_loss = train_running_loss / (idx + 1) \
                                            / self.train_dataloader.batch_size
            if not train_loss0:
                if not np.isnan(train_loss):
                    train_loss0 = train_loss

            self.train_history['train losses'].append(train_loss)
            train_acc = self.model_env.accuracy(
                    self.model_env.cumulate(train_current))
            self.train_history['train acc'].append(train_acc['acc'])

            self.model_env.model.eval()
            with torch.no_grad():
                val_running_loss = 0               

                for idx, batch in enumerate(self.val_dataloader):
                    print_progress('validation', epochs, idx, self.val_dataloader)
                    
                    loss = self.model_env(batch)
                    val_current.append(self.model_env.current)
                    val_running_loss += loss.item()           
              
            val_loss = val_running_loss / (idx + 1) \
                                                / self.val_dataloader.batch_size
            if not val_loss0:
                if not np.isnan(val_loss):
                    val_loss0 = val_loss

            self.train_history['val losses'].append(val_loss)
            val_acc = self.model_env.accuracy(
                                        self.model_env.cumulate(val_current))
            self.train_history['val acc'].append(val_acc['acc'])
            
            if best is None: 
                best = val_acc['accuracy']
            if (val_acc['accuracy'] / (abs(best) + 1e-12) - 1) \
                                                    > best_thd:
                best = val_acc['accuracy']
                if self.project_shell is not None:
                    self.project_shell.save_model(
                        best=True,
                        verbose=False)
                    
            # stop if `ctr_value` is clearly positive:
            if self.project_shell.stopper.stop(
                # train loss is much smaller then val loss:
                    val_loss / val_loss0 - train_loss / train_loss0):
                if self.project_shell.stop():
                    break

            first_epoch = False

    def save_model(self, file_path):
        self.model_env.model.save_pretrained(file_path)
