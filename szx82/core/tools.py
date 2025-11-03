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

    def on_key_submit(self, key):
        if key in [_[1] for _ in self.keys]:
            self.key = key
            print(f' Key "{self.key.decode()}" pressed, wait.')
        else:
            self.on_key_pressed_done

    def on_key_pressed_done(self):
        pass
 
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
            self.on_key_pressed_done()
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
                key = msvcrt.getch()
                if key in [_[1] for _ in self.keys]:
                    self.key = key
                    print(f' Key "{self.key.decode()}" pressed, wait.')

    def print_menu(self):
        msg = '\n'
        for k in self.keys:
            msg += f'{k[0]}: {k[1].decode()}\n'
        print(msg)
             