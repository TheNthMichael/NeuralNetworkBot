import numpy as np
import os
import nnmodel
from datetime import datetime
from gameDataLoader import *

dirname = os.path.dirname(__file__)

def train(batch_size, eta):
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # For getting the sizes, we will read the first sample, then ignore it.
    myData = os.path.join(dirname, "TestData", "dataset_tps_akina10k")
    inputsize, outputsize = getSizes(myData)
    print(f"InputSize: {inputsize}\nOutputSize: {outputsize}")
    net = nnmodel.Network([inputsize, 40, 10, 40, outputsize])
    for data_set, inputsize, outputsize in loader(myData, batch_size):
        print("Starting new batch")
        net.SGD(data_set, 10, batch_size, eta, test_data=data_set)
    
    nnmodel.save_model(f"SavedModel{datetime.now().strftime('%Y%m%d%H%M%S')}", net)