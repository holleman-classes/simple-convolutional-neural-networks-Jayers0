import numpy as np

# Function to calculate the number of parameters in a convolutional layer
def conv2d_params(input_channels, output_channels, kernel_size, stride=1, padding=0):
    return (input_channels * kernel_size[0] * kernel_size[1] + 1) * output_channels

# Function to calculate the number of parameters in a dense (fully connected) layer
def dense_params(input_units, output_units):
    return (input_units + 1) * output_units

# Define the structure of the CNN
layers = [
    {"type": "conv2d", "filters": 32, "kernel_size": (3, 3), "stride": 2, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 64, "kernel_size": (3, 3), "stride": 2, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 128, "kernel_size": (3, 3), "stride": 2, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 128, "kernel_size": (3, 3), "stride": 1, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 128, "kernel_size": (3, 3), "stride": 1, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 128, "kernel_size": (3, 3), "stride": 1, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "conv2d", "filters": 128, "kernel_size": (3, 3), "stride": 1, "padding": "same"},
    {"type": "batchnorm"},
    {"type": "maxpooling", "pool_size": (4, 4), "stride": (4, 4)},
    {"type": "flatten"},
    {"type": "dense", "units": 128},
    {"type": "batchnorm"},
    {"type": "dense", "units": 10}
]

# Initialize variables to keep track of input/output sizes
input_channels = 3
input_size = 32

# Loop through the layers to calculate parameters
total_params = 0
for layer in layers:
    if layer["type"] == "conv2d":
        filters = layer["filters"]
        kernel_size = layer["kernel_size"]
        stride = layer.get("stride", 1)
        padding = layer["padding"]
        
        output_size = input_size // stride
        params = conv2d_params(input_channels, filters, kernel_size)
        
        total_params += params
        input_channels = filters
        input_size = output_size
        
    elif layer["type"] == "dense":
        units = layer["units"]
        params = dense_params(input_size * input_size * input_channels, units)
        total_params += params
        
    elif layer["type"] == "flatten":
        input_channels = input_channels * input_size * input_size
        
    elif layer["type"] == "maxpooling":
        pool_size = layer["pool_size"]
        stride = layer["stride"]
        output_size = input_size // stride[0]
        input_size = output_size
        
    elif layer["type"] == "batchnorm":
        # BatchNorm has no parameters
        pass

print("Total number of parameters in the CNN model:", total_params)
