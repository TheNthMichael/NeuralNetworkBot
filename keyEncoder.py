from pynput.keyboard import Key

class MyKey:
    def __init__(self):
        self.functionA = lambda func: func('a')
        self.functionW = lambda func: func('w')
        self.functionS = lambda func: func('s')
        self.functionD = lambda func: func('d')
        self.functionE = lambda func: func('e')
        self.functionQ = lambda func: func('q')
        self.functionSHIFT = lambda func: func(Key.shift)