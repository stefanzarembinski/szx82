from os import path, makedirs, listdir
import time
import numpy as np
import pickle
import zlib
from matplotlib import pyplot as plt
import szx81.core.core_ as co
from szx81.core.helper import KeyPressed

class ProjectShell:
    def __init__(
            self,
            data_object,
            model_shell,
            labels,
            store_dir, 
            file_name=None, 
            save=True):

        self.data_object = data_object
        self.ms = model_shell
        self.ms.transformer_shell = self
        self.can_save = save
        if not self.can_save:
            print('\nPARAMETR "save" ma wartość "False"! ################\n')
        
        self.file_name_first = file_name
        self.store_dir = path.join(store_dir, self.file_name())

        self.label = {}
        for label in labels:
            self.add_label(label)
        self.add_label(model_shell.label)
        self.key_pressed = KeyPressed(locals())
        self.key_pressed.keys = [            
            [
                'key to STOP training',
                b's',
# must be boolean `retval` returned:
                ''' 
while msvcrt.kbhit():
    msvcrt.getch()
yes_or_no = input(
    '"Y" for stopping, anything else for continuing: '
)
if yes_or_no == "Y":
    self.save(verbose=True)
    retval = True
''',
            ],
            [
                'key to save current training and print report',
                b'r',
                [
                    (self.save, {'verbose': True}),
                ],
            ],
            [
                'key to save torch model',
                b'm',
                (self.save_model, {'verbose': True}),
            ],
            [
                'key to plot losses',
                b'p',
                self.plot_training,
            ],
            [
                'key to set weight',
                 b'w',
                 '''
while True:
    weight = self.ms.model_env.criterion.weight
    if weight is not None:
        weight = [round(_, 2) for _ in weight.tolist()]

        weight = input(f"""

Weight is {weight}.
input weight or nothing to abort: """
                    )
        if not weight:
            break
        try:
            weight = [float(_) for _ in weight.split(',')]
        except Exception as ex:
            print(ex)
            continue
        error = self.ms.model_env.set_weight(weight)
        if error is None:
            break
'''
            ]
        ]

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
        self.key_pressed.print_manu()          
        self.ms.train()

    @staticmethod
    def str_label(label):
        return '\n'.join('{0}: {1}'.format(k, v) for k, v in label.items())
    
    def add_label(self, label):
        for k, v in label.items():
            if k in self.label:
                assert self.label[k] != v,\
                    f'"{k}" value is not unique, either {self.label} or {[k]})'
            self.label[k] = v

    def add_to_label(self, k, v):
        if k in self.label:
                assert self.label[k] != v,\
                    f'"{k}" value is not unique, either {self.label} or {[k]})'
        self.label[k] = v

    def __str__(self):
        return self.str_label(self.label)

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
                    exit()
            elif not path.exists(self.store_dir):
                makedirs(self.store_dir, exist_ok=True)

    def save(self, file_path=None, prep=None, verbose=True):
        self.save_model() # just update result records of the model 
        if prep is not None:
            file_path = prep + '_' + file_path
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


    
