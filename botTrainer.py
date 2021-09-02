from cv2 import drawMatchesKnn
import numpy as np
import os
import nnmodel
from datetime import datetime
from dataLoader import DataLoader

def train(path, batch_size, eta, chunk_size:int=None, model_save_name:str=None):
    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # For getting the sizes, we will read the first sample, then ignore it.
    if eta is None:
        eta = 0.03
    if batch_size is None:
        batch_size = 100
    myData = os.path.join(path)
    dataLoader = DataLoader(myData, normalize=True, chunk_size=chunk_size)
    """testData = DataLoader(myData, normalize=True, chunk_size=chunk_size)
    for dataset in testData:
        testData = dataset
        break"""

    inputsize = dataLoader.input_size
    outputsize = dataLoader.output_size
    modelArch = [inputsize, 512, 512, 256, 256, 128,256,256, 40, outputsize]
    print(f"InputSize: {inputsize}\nOutputSize: {outputsize}")
    net = nnmodel.Network(modelArch, dataLoader._norm_attrs)
    for data_set in dataLoader:
        print(f"Starting new batch of size {len(list(data_set))}.")
        net.SGD(data_set, 10, batch_size, eta, evaluation_data=None)

    if not model_save_name:
        nnmodel.save_model(f"SavedModel{datetime.now().strftime('%Y%m%d%H%M%S')}", net)
    else:
        nnmodel.save_model(model_save_name, net)