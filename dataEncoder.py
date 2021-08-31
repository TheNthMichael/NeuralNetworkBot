import os
import numpy as np
from pynput.keyboard import Key

"""Convert pynput key code into an index for an array."""
KEY_TO_CODE_MAP = {
    "'w'": 0,
    "'a'": 1,
    "'s'": 2,
    "'d'": 3,
    "'q'": 4,
    "'e'": 5,
    "'w'": 0,
    "key.shift": 6,
    "key.space": 7
}

"""Convert array index to pynput key code."""
CODE_TO_KEY_MAP = {
    0 : 'w',
    1 : 'a',
    2 : 's',
    3 : 'd',
    4 : 'q',
    5 : 'e',
    6 : Key.shift,
    7 : Key.space
}

"""Maps the passed key to its corresponding array code.

Returns None if the key is not in the mapping."""
def map_key_to_code(key):
    key_lower = str(key).lower()
    if key_lower in KEY_TO_CODE_MAP:
        return KEY_TO_CODE_MAP[key_lower]
    return None

"""Maps the passed code to its corresponding pynput key.

Returns None if the code is not in the mapping."""
def map_code_to_key(code):
    if code in CODE_TO_KEY_MAP:
        return CODE_TO_KEY_MAP[code]
    return None

"""Transforms the value x from the input range to the output range."""
def linmap(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

"""Clamps the value x to the minv and maxv range."""
def clamp(x, minv, maxv):
    assert(minv <= maxv)
    return min(max(x, minv), maxv)

def find_mouse_range(data_folder):
    xMouseMin, xMouseMax, yMouseMin, yMouseMax = None
    for filename in os.listdir(data_folder):
        filename = os.path.join(data_folder, filename)
        file = np.load(filename)
        mouse = file['mouse']
        xMouseMin = min(xMouseMin, mouse[0])
        xMouseMax = max(xMouseMax, mouse[0])
        yMouseMin = min(yMouseMin, mouse[1])
        yMouseMax = max(yMouseMax, mouse[1])
    return xMouseMin, xMouseMax, yMouseMin, yMouseMax

