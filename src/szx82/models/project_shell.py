import sys
import time
import timeit
from os import path, makedirs, listdir
import time
import numpy as np
import pickle
import zlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

class Stopper:
    def __init__(self, min_ep_count=20, timeout=60, 
                 tail=20, threshold=0.2):
        self.min_ep_count = min_ep_count
        self.timeout = timeout * 60
        self.tail = tail
        self.threshold = threshold
        self.start_time = time.time()
        self.ctr_value = []
    
    def stop(self, ctr_value, scale=1):
        # if len(self.ctr_value) > 3:
        #     return True
        # return `True` if  `ctr_value` is positive
        self.ctr_value.append(ctr_value)
        if len(self.ctr_value) < self.min_ep_count:
            return False        
        return np.mean(self.ctr_value[-self.tail:]) > scale * self.threshold

class ProjectShell:
    def __init__(
            self, 
            model_shell,
            store_dir, 
            file_name=None,
            stop_thd=0.1,
            save=True):

        self.ms = model_shell
        self.ms.project_shell = self
        self.can_save = save
        if not self.can_save:
            print('\nPARAMETR "save" ma wartość "False"! ################\n')
        self.ok_run = False
        self.file_name_first = file_name
        self.store_dir = path.join(store_dir, self.file_name())
        self.start = timeit.default_timer()
        self.stopper = Stopper(stop_thd)

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
            time.sleep(5)
        except:
            print('ERROR in "plot_training"!')

    def train(self):
        if self.ok_run:      
            self.ms.train()
    
    def stop(self):
        print(
        f"Training Time: {timeit.default_timer() - self.start:.2f} s")        
        self.plot_training()

        return True

    def file_name(self):
        if self.file_name_first is not None:
            return self.ms.model_env.FILE_PREFIX + '_' + self.file_name_first

        file_name = f'{type(self.ms).__name__}_{zlib.adler32((self.__str__() + self.ms.__str__()) .encode())}'

        return self.ms.model_env.FILE_PREFIX + file_name
              
    def file_exists(self, force=False):
        self.ok_run = True
        print(f'''
project dir: {path.normpath(self.store_dir)}
project name: {self.file_name()}
''')    
        if self.can_save:
            if path.exists(self.store_dir) and listdir(self.store_dir):
                if not force:
                    yes_or_no = input(
                        f'\r "Y" for overwriting existing file in dir {path.normpath(self.store_dir)}, anything else for aborting: the process: '
                    )
                    if yes_or_no != "Y":
                        self.ok_run = False
            elif not path.exists(self.store_dir):
                makedirs(self.store_dir, exist_ok=True)

    def get_model_file(self, best=False):
        if best:
            return path.join(
            self.store_dir, self.file_name() + '_' + 'bst' + '_' + '.pt')
        return  path.join(self.store_dir, self.file_name() + '.pt')



    
