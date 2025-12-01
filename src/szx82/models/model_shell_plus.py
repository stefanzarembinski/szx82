import numpy as np
from tqdm import tqdm
import timeit
import torch
from szx82.models.model_shell import ModelShell

class ModelShellPlus(ModelShell):
    def __init__(
            self, 
            train_dataloader, 
            val_dataloader, 
            ModelClass, 
            config_or_path, 
            lr=0.001, 
            weight_decay=0, 
            betas=(0.9, 0.999), 
            warmup_steps=10000, 
            **kwargs):
        super().__init__(train_dataloader, val_dataloader, ModelClass, config_or_path, lr, weight_decay, betas, warmup_steps, **kwargs)

    def train(self):
        start = timeit.default_timer()
        train_loss = None
        train_loss0 = None
        
        val_loss = None
        val_loss0 = None
        
        best = None
        best_thd = 1e-4 
        ncols=80
        train_acc = None
        val_acc = None
        self.model_env.transformer_shell = self.transformer_shell

        epochs = tqdm(desc='epochs', ncols=ncols)
        epochs.update(1)
        first_epoch = True
        
        self.model_env.final_adj()

        while True:
            epochs.update(1)
            if train_loss0 and val_loss0:
                epochs.set_description(f'loss trn,val:{train_loss / train_loss0:4.2f},{val_loss / val_loss0:4.2f} best: { best:0.2f}')

            self.model_env.model.train()
            train_current = []
            val_current = []            

            train_running_loss = 0 
            train_tqdm = tqdm(self.train_dataloader, leave=False, ncols=ncols)
            val_tqdm = tqdm(self.val_dataloader, leave=False, ncols=ncols)

            if self.transformer_shell is not None:
                 if self.transformer_shell.key_pressed.key_pressed():
                    break

            if train_loss is not None:
                train_tqdm.set_description(
                                    f'train {train_acc["msg"]}')     
                val_tqdm.set_description(f'val. {val_acc["msg"]}') 
            
            for idx, batch in enumerate(train_tqdm):
                self.transformer_shell.key_pressed.read_key_pressed()

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
            train_current = self.model_env.cumulate(train_current)
            train_acc = self.model_env.accuracy(train_current)
            self.train_history['train acc'].append(train_acc['acc'])
            self.train_results = train_current

            self.model_env.model.eval()
            with torch.no_grad():
                val_running_loss = 0               

                for idx, batch in enumerate(val_tqdm):
                    self.transformer_shell.key_pressed.read_key_pressed()
                    
                    loss = self.model_env(batch)
                    val_current.append(self.model_env.current)
                    val_running_loss += loss.item()           
              
            val_loss = val_running_loss / (idx + 1) \
                                                / self.val_dataloader.batch_size
            if not val_loss0:
                if not np.isnan(val_loss):
                    val_loss0 = val_loss

            self.train_history['val losses'].append(val_loss)
            val_current = self.model_env.cumulate(val_current)
            val_acc = self.model_env.accuracy(val_current)
            self.train_history['val acc'].append(val_acc['acc'])
            self.val_results = val_current
            
            if best is None:
                best = val_acc['best']
            if (val_acc['best'] / (abs(best) + 1e-12) - 1) \
                                                    > best_thd:
                best = val_acc['best']
                if self.transformer_shell is not None:
                    self.transformer_shell.save_model(
                        best=True,
                        verbose=False)

            first_epoch = False

        stop = timeit.default_timer()
        print(f"Training Time: {stop-start:.2f} s")

