import sys
from os import path, makedirs, listdir
import time
import numpy as np
import pickle
import zlib
from matplotlib import pyplot as plt

class ProjectShell:
    def __init__(
            self,
            model_shell,
            store_dir, 
            file_name=None, 
            save=True):

        self.ms = model_shell
        self.ms.transformer_shell = self
        self.can_save = save
        if not self.can_save:
            print('\nPARAMETR "save" ma wartość "False"! ################\n')
        
        self.file_name_first = file_name
        self.store_dir = path.join(store_dir, self.file_name())

    def print_result(self):
        pass # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def plot_training(self, figsize=(7, 5), dpi=100):
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

    def stop(self):
        """On stopping the current process.""" 
        self.save_self(verbose=True)

    def train(self):      
        self.ms.train()

    def file_name(self):
        if self.file_name_first is not None:
            return self.ms.model_env.FILE_PREFIX + '_' + self.file_name_first

        file_name = f'{type(self.ms).__name__}_{zlib.adler32((self.__str__() + self.ms.__str__()) .encode())}'

        return self.ms.model_env.FILE_PREFIX + file_name
              
    def file_exists(self):
        print(f'''
project dir: {path.normpath(self.store_dir)}
project name: {self.file_name()}
''')    
        if self.can_save:
            if path.exists(self.store_dir) and listdir(self.store_dir):
                yes_or_no = input(
                    f'\r "Y" for overwriting existing file in dir {path.normpath(self.store_dir)}, anything else for aborting: the process: '
                )
                if yes_or_no != "Y":
                    sys.exit()
            elif not path.exists(self.store_dir):
                makedirs(self.store_dir, exist_ok=True)

    def save(self, file_path=None, verbose=True):
        self.save_model() # just update result records of the model 
        if file_path is None:
            file_path = path.join(self.store_dir, self.file_name() + '.pkl')
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    def save_self(self, verbose=False):
        try:
            if self.can_save:
                self.save(verbose=verbose)
        except Exception as ex:
            print(ex)

    def get_model_file(self, best=False):
        if best:
            return path.join(
            self.store_dir, self.file_name() + '_' + 'bst' + '_' + '.pt')
        return  path.join(self.store_dir, self.file_name() + '.pt')

    def save_model(self, best=False, verbose=True):           
        try:
            self.ms.save_model(file_path := self.get_model_file(best))
            if verbose:
                print(f'''Model saved:
{file_path}
''')
        except Exception as ex:
            print(ex)


    
