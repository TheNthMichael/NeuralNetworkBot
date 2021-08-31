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
            self.map_mouse_to_sigmoid = None

    """Creates a new instance of the data loader.
    
    normalize is a flag which when set will normalize all data points to a range 0, 1.
    some data points such as mouse dp's require normalization so that they can be
    un-normalized afterward to retrieve a normal mouse value as all values get squished
    into the (0, 1) range of the sigmoid function regardless. The reason to normalize all
    values is to ensure one value doesn't oversaturate others."""
    def __init__(self, path, normalize=False, chunk_size: int=None) -> None:
        self.path = path
        self.normalize = normalize
        self.chunk_size = chunk_size
        # Our model will hold data for approx 12 frames of a key being held down.
        # after this point, it will just be flagged as "key has been down as long
        # as I've known."
        self.look_back_time = 12
        # As x approaches infinity norm_map approaches 1.
        self.norm_map = lambda x: 1 - math.exp(-x/self.look_back_time)
        # As x approaches infinity invnorm_map approaches 0.
        self.invnorm_map = lambda x: math.exp(-x/self.look_back_time)

        self.training_set = os.listdir(self.path)
        self.input_size, self.output_size = self.get_sizes()
        if self.normalize:
            self.get_normalize_attributes()

    def get_normalize_attributes(self):
        self._norm_attrs = self.NormalizedAttributes()
        self._norm_attrs.imageValMin = 0
        self._norm_attrs.imageValMax = 255
        for filename in os.listdir(self.path):
            filename = os.path.join(self.path, filename)
            print(f"Opened {filename}.")
            file = np.load(filename)
            mouse = file['mouse']
            self._norm_attrs.xMouseMin = min(self._norm_attrs.xMouseMin, mouse[0])
            self._norm_attrs.xMouseMax = max(self._norm_attrs.xMouseMax, mouse[0])
            self._norm_attrs.yMouseMin = min(self._norm_attrs.yMouseMin, mouse[1])
            self._norm_attrs.yMouseMax = max(self._norm_attrs.yMouseMax, mouse[1])
        self._norm_attrs.map_mouse_to_sigmoid = lambda x, y: (dataEncoder.linmap(x,\
           self._norm_attrs.xMouseMin, self._norm_attrs.xMouseMax, 0, 1),\
               dataEncoder.linmap(y, self._norm_attrs.yMouseMin, self._norm_attrs.yMouseMax, 0, 1))
        

    """Returns the expected sizes for input and output of
    the data set inside the given path. Estimates this from
    the first training example.

    Returns input_size, output_size."""
    def get_sizes(self):
        for filename in os.listdir(self.path):
            filename = os.path.join(self.path, filename)
            print(f"Opened {filename}.")
            file = np.load(filename)
            keys = file['keys']
            mouse = file['mouse']
            input = file['image']
            keysframecount = file['keysframecount']
            input = np.append(input, keysframecount)
            input = input.reshape([input.shape[0], 1])
            input_size = input.shape[0]
            output = np.append(keys, mouse)
            output = np.transpose(output)
            print(f"SHAPE: {output.shape}\t{input.shape}")
            output_size = output.shape[0]
            return input_size, output_size
    
    def get_training_set(self, chunk_size: int=None):
        assert(chunk_size is None or chunk_size > 0)
        trainingInputs = []
        trainingOutputs = []
        endpoint = min(len(self.training_set), chunk_size)
        training_subset = self.training_set[:endpoint]
        self.training_set = self.training_set[endpoint:]
        for filename in training_subset:
            filename = os.path.join(self.path, filename)
            file = np.load(filename)
            keys = file['keys']
            mouse = file['mouse']
            input = file['image']
            keysframecount = file['keysframecount']
            # Map the history of keys being pressed for some
            # number of frames into a value bounded (0,1).
            # Also map the grayscale image to values bounded (0, 1).
            if self.normalize:
                keysframecount = self.norm_map(keysframecount)
                input = input / 255
                mouse[0], mouse[1] = self._norm_attrs.map_mouse_to_sigmoid(mouse[0], mouse[1])
            input = np.append(input, keysframecount)
            input = input.reshape([input.shape[0], 1])
            output = np.append(keys, mouse)
            output = output.reshape([output.shape[0], 1])
            trainingInputs.append(input)
            trainingOutputs.append(output)
        data_set = zip(trainingInputs, trainingOutputs)
        return data_set
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.training_set) > 0:
            return self.get_training_set(self.chunk_size)
        raise StopIteration
            

"""Loads the data into zip object containing the input training
examples and the expected output training results.
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
    return loader"""

"""The loader for image and input data for the model.
Format for input and output data
input: [0.5, 0.1, ..., 0.9] Range = (0, 1) Image Data
output: [Keys=(0.23, 0.99, 0.12, 0.98, 0.1, 0.76, 0.99),
Mouse=(0.55, 0.3)] Range = (0, 1) Keyboard discrete states
and Mouse real valued states.
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
        yield data_set, input_size, output_size"""