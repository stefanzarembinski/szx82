import sys
import time
import timeit
from os import path, makedirs, listdir
import time
import numpy as np
import json
import zlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class Stopper:
    def __init__(self, min_ep_count=20, timeout=60, 
                 tail=20, threshold=0.02 ):
        self.min_ep_count = min_ep_count
        self.timeout = timeout * 60
        self.tail = tail
        self.threshold = threshold
        self.start_time = time.time()
        self.ctr_value = []
    
    def stop(
            self, train_loss, val_loss, new_best):
        # return `True` if  `ctr_value` is positive
        ctr_value = 1 - train_loss / val_loss
        self.ctr_value.append(ctr_value)
        if (epochs := len(self.ctr_value)) < self.min_ep_count:
            return False
        no_changes = (1 - new_best / epochs) > 0.2
        return (ctr_value > self.threshold) or no_changes

class ProjectShell:
    MODEL_PT = 'model'
    def __init__(
            self, 
            model_shell,
            store_dir, 
            name_prep=None, 
            stop_thd=0.02,
            save=True):

        self.ms = model_shell
        self.ms.project_shell = self
        self.can_save = save
        if not self.can_save:
            print('\nPARAMETR "save" ma wartość "False"! ################\n')
        self.ok_run = False
        self.name_prep = name_prep
        self.store_dir = path.join(store_dir, self.project_name())
        self.start = timeit.default_timer()
        self.stopper = Stopper(threshold=stop_thd)
        self.project_file = 'args.json'

    def print_result(self):
        pass # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def plot_training(self, figsize=(7, 3), dpi=100):
        train_history = self.ms.train_history

        def accuracy(acc):
            acc = train_history[acc]
            labels = list(acc[0].keys())
            values = {k: [] for k in labels}
            for _ in acc:
                for k, v in _.items():
                    values[k].append(v)
            return values, labels
        try:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            label = 'train losses'
            values = np.array(train_history[label]) / train_history[label][0] 
            ax.plot(values, label=label + ' norm.')
            label = 'val losses'
            values = np.array(train_history[label]) / train_history[label][0]
            ax.plot(values, label=label + ' norm.')
            label = 'train acc'
            values, labels = accuracy(label)
            for _ in labels:
                ax.plot(values[_], label=label + ' ' + _)
            label = 'val acc'
            values, labels = accuracy(label)
            for _ in labels:
                ax.plot(values[_], label=label + ' ' + _)

            plt.legend()
            plt.show()
        except:
            print('ERROR in "plot_training"!')

    def train(self):
        if self.ok_run:      
            self.ms.train()
    
    def stop(self):
        print(
        f"Training Time: {timeit.default_timer() - self.start:.2f} s")        
        self.plot_training()
        time.sleep(10)
        return True 
 
    def project_name(self):
        if self.name_prep is not None:
            return self.ms.model_env.FILE_PREFIX \
                + ('_' + self.name_prep) if self.name_prep else ''

        file_name = f'{type(self.ms).__name__}_{zlib.adler32((self.__str__() + self.ms.__str__()) .encode())}'
 
        return self.ms.model_env.FILE_PREFIX + file_name
              
    def file_exists(self, force=False):
        self.ok_run = True
        print(f'''
project dir: {path.normpath(self.store_dir)}
project name: {self.project_name()}
''')    
        if self.can_save:
            if path.exists(self.store_dir) and listdir(self.store_dir):
                if not force:
                    yes_or_no = input(
                        f'\r "Y" for overwriting existing file in dir {path.normpath(self.store_dir)}, anything else to abort: '
                    )
                    if yes_or_no != "Y":
                        self.ok_run = False
            elif not path.exists(self.store_dir):
                makedirs(self.store_dir, exist_ok=True)

    def get_model_file(self, best=False): 
        if best:
            return path.join(
            self.store_dir, self.MODEL_PT + '_' + 'bst' + '_' + '.pt')
        return  path.join(self.store_dir, self.MODEL_PT + '.pt')
    
    def save_model(self, best=True, verbose=False): 
        file_path = self.get_model_file(best=best)
        self.ms.save_model(file_path)
        if verbose:
            print(f'''model saved: {file_path}''')

    def save_project(self, project_args):
        with open(path.join(self.store_dir, self.project_file), "w") \
                                                            as json_file:
            json.dump({k: repr(v) for k, v in project_args.items()}, 
                      json_file, sort_keys=True, indent=4,)



    
