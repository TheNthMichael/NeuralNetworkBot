import os
import math
import numpy as np
import dataEncoder

class DataLoader:
    class NormalizedAttributes:
        def __init__(self) -> None:
            self.xMouseMin = None
            self.xMouseMax = None
            self.yMouseMin = None
            self.yMouseMax = None
            self.imageValMin = None
            self.imageValMax = None

    """Creates a new instance of the data loader.
    
    normalize is a flag which when set will normalize all data points to a range 0, 1.
    some data points such as mouse dp's require normalization so that they can be
    un-normalized afterward to retrieve a normal mouse value as all values get squished
    into the (0, 1) range of the sigmoid function regardless. The reason to normalize all
    values is to ensure one value doesn't oversaturate others."""
    def __init__(self, path, normalize=False) -> None:
        self.path = path
        # Our model will hold data for approx 12 frames of a key being held down.
        # after this point, it will just be flagged as "key has been down as long
        # as I've known."
        self.look_back_time = 12
        # As x approaches infinity norm_map approaches 1.
        self.norm_map = lambda x: 1 - math.exp(-x/self.look_back_time)
        # As x approaches infinity invnorm_map approaches 0.
        self.invnorm_map = lambda x: math.exp(-x/self.look_back_time)

        if normalize:
            self.get_normalize_attributes()

    def get_normalize_attributes(self):
        self._norm_attrs = self.NormalizedAttributes()
        self._norm_attrs.imageValMin = 0
        self._norm_attrs.imageValMax = 255
        for filename in os.listdir(self.path):
            filename = os.path.join(self.path, filename)
            print(f"Opened {filename}.")
            file = np.load(filename)
            keys = file['keys']
            mouse = file['mouse']
            input = file['image']
        

"""Returns the expected sizes for input and output of
the data set inside the given path. Estimates this from
the first training example.

Returns input_size, output_size."""
def getSizes(path :str):
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        print(f"Opened {filename}.")
        file = np.load(filename)
        keys = file['keys']
        mouse = file['mouse']
        input = file['image']
        input_size = input.shape[0]
        output = np.append(keys, mouse)
        output = np.transpose(output)
        print(f"SHAPE: {output.shape}\t{input.shape}")
        output_size = output.shape[0]
        return input_size, output_size

"""Loads the data into zip object containing the input training
examples and the expected output training results."""
def get_loader(path :str, batchSize :int):
    def loader():
        trainingInputs = []
        trainingOutputs = []
        xmin, xmax, ymin, ymax = dataEncoder.find_mouse_range(path)
        batchCount = 0
        for filename in os.listdir(path):
            filename = os.path.join(path, filename)
            file = np.load(filename)
            keys = file['keys']
            mouse = file['mouse']
            mouse[0], mouse[1] = dataEncoder.map_to_sigmoid(mouse[0], mouse[1])
            input = file['image']
            output = np.append(keys, mouse)
            output = output.reshape([output.shape[0], 1])
            trainingInputs.append(input)
            trainingOutputs.append(output)
            if batchCount > batchSize:
                batchCount = 0
                data_set = zip(trainingInputs[:], trainingOutputs[:])
                trainingInputs.clear()
                trainingOutputs.clear()
                yield data_set
        if len(trainingInputs) > 0:
            data_set = zip(trainingInputs[:], trainingOutputs[:])
            trainingInputs.clear()
            trainingOutputs.clear()
            yield data_set
    return loader

"""The loader for image and input data for the model.
Format for input and output data
input: [0.5, 0.1, ..., 0.9] Range = (0, 1) Image Data
output: [Keys=(0.23, 0.99, 0.12, 0.98, 0.1, 0.76, 0.99), Mouse=(0.55, 0.3)] Range = (0, 1) Keyboard discrete states and Mouse real valued states.
"""
def loader(path :str, batchSize :int):
    trainingInputs = []
    trainingOutputs = []
    encoder = DataEncoderAndDecoder()
    encoder.find_mouse_range(path)
    input_size = 0
    output_size = 0
    batchCount = 0
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        file = np.load(filename)
        keys = file['keys']
        mouse = file['mouse']
        mouse[0], mouse[1] = encoder.map_to_sigmoid(mouse[0], mouse[1])
        input = file['image']
        input_size = input.shape[0]
        output = np.append(keys, mouse)
        output = output.reshape([output.shape[0], 1])
        output_size = output.shape[0]
        trainingInputs.append(input)
        trainingOutputs.append(output)
        if batchCount > batchSize:
            batchCount = 0
            data_set = zip(trainingInputs[:], trainingOutputs[:])
            trainingInputs.clear()
            trainingOutputs.clear()
            yield data_set, input_size, output_size
    if len(trainingInputs) > 0:
        data_set = zip(trainingInputs[:], trainingOutputs[:])
        trainingInputs.clear()
        trainingOutputs.clear()
        yield data_set, input_size, output_size