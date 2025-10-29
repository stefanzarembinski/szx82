import time
import traceback
import msvcrt 
from collections.abc import Iterable
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def slice(start_stop, data):
    start_stop = list(start_stop)

    if start_stop[0] is not None \
        and start_stop[0] !=0 and start_stop[0] < 1: 
        start_stop[0] = int(len(data) * start_stop[0])
    if start_stop[1] is not None \
        and start_stop[1] !=0 and start_stop[1] < 1:
        start_stop[1] = int(len(data) * start_stop[1])

    return tuple(start_stop)

class KeyPressed:
    def __init__(self, locals=None):
        self.locals = locals
        self.key = None
        self.keys = [
            [
                'key to save',
                b's',
                'self.save()'
            ],
            [ 

            ]
        ]

    def key_pressed(self, locals=None):
        try:
            retval = False
            key = self.key
            if msvcrt.kbhit():
                key = msvcrt.getch()
            if (key is not None) and (key in [_[1] for _ in self.keys]):
                print(f' Processing "{key.decode()}" pressed...')
                for k in self.keys:
                    if key == k[1]: 
                        if isinstance(k[2], str):
                            if locals is None:
                                locals = self.locals
                            locals['retval'] = retval
                            
                            exec(k[2], None, locals)
                            retval = locals['retval']
                        elif isinstance(k[2], Iterable):
                            if isinstance(k[2][0], Iterable):
                                for _ in k[2]:
                                    if isinstance(_, Iterable):
                                        _[0](**_[1])
                                    else:
                                        _()
                            else:
                                k[2][0](**k[2][1])
                        else:
                            retval = k[2]()
                        break

                time.sleep(2)
            self.key = None
            return False if retval is None or not retval else True
        except Exception as ex:
            print(f'''
ERROR in the `key_pressed` function.
Error message is 
{ex}
{traceback.print_exc()}
''')
    def read_key_pressed(self):
        if msvcrt.kbhit():
                self.key = msvcrt.getch()
                if self.key in [_[1] for _ in self.keys]:
                    print(f' Key "{self.key.decode()}" pressed, wait.')

    def print_manu(self):
        msg = '\n'
        for k in self.keys:
            msg += f'{k[0]}: {k[1].decode()}\n'
        print(msg)

    def __init__(
            self,
            idxs,
            forex_value,
            futures,
            direction, count, 
            ):
        self.idxs = idxs
        self.value = forex_value
        self.futures = futures
        self.direction = direction
        self.count = count
        self.slider = None
        self.val_plot = None

        self.fig = plt.figure('Future', figsize=(6, 4), dpi=90,)
        self.ax = self.fig.add_axes([0.1, 0.2, 0.9, 0.8]) 
        plt.subplots_adjust(bottom=0.25)
        plt.rcParams['toolbar'] = 'None'

        self.update(None)
        ax_slider = plt.axes([0.1, 0.1, 0.9, 0.03])
        self.slider = Slider(
             ax_slider, 'idxs', 0, len(idxs) - count, valinit=0)
        self.slider.on_changed(self.update)
        
        self.fig.legend()
        plt.show()

    def update(self, val):
        start = 0 if self.slider is None else int(self.slider.val)
        stop = min(start + self.count, len(self.idxs))
        idxs_ = self.idxs[start: stop]
        value_ = self.value[start:stop]

        if self.val_plot is not None:
            self.ax.cla()

        for k, v in self.futures.items():
            mask = (v[start:stop] == self.direction)
            self.ax.plot(idxs_[mask], value_[mask], 'o', ms=9,
                        alpha=0.3, label=k)
        self.val_plot, = self.ax.plot(
            idxs_, value_, linewidth=0.5, label='forex')
        self.ax.set_xlim(idxs_.min(), idxs_.max())
        self.ax.set_ylim(
                    value_.min() - 2e-4, value_.min() + 40e-4)               