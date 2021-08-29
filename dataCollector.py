import numpy as np
import sys
import cv2
import time
import os
from datetime import datetime
from mss import mss
from PIL import Image
import pynput
from pynput.keyboard import Key, Listener as KeyListener
from pynput.mouse import Listener as MouseListener
from dataEncoding import *

def on_press_handler(key):
    if not str(key).lower() in keys:
        keys.append(str(key).lower())
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        print('special key {0} pressed'.format(key))

class TrainingExample:
    def __init__(self, prev_image, prev_inputs, image, \
        inputs, input_frame_count):
        self.prev_image = prev_image
        self.prev_inputs = prev_inputs
        self.image = image
        self.inputs = inputs
        self.input_frame_count = input_frame_count

class DataCollector:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path
        if self.dataset_path == None:
            dirname = os.path.dirname(__file__)
            testdir = os.path.join(dirname, "TestData")
            if not os.path.exists(testdir):
                os.mkdir(testdir)
            self.dataset_path = os.path.join(testdir, \
                f"dataset{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.mkdir(self.dataset_path)

    def run(self):
        pass