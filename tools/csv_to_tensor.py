"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
This file takes a csv file (e.g. written by numpy) and puts the values
 as an array into a TensorProto that is stored as a file.
Such a TensorProto is required to run MiniONN as a client.
Obviously, there are many ways to give such a TensorProto as input but 
 this example file should be enough to get the idea of how to work with TensorProto.
"""

import onnx
import struct
import numpy as np

filename = "array.txt"
delimiter = ","
tensor_name = "1"

# Load values and convert to list
values = np.loadtxt(filename, delimiter=delimiter)
values_list = values.flatten().tolist()

# Pack the input into raw bytes
values_raw = struct.pack('%sf' % len(values_list), *values_list)

# export the raw data to a tensor proto. 
#  We use FLOAT type here but pack it in bytes
t_type = onnx.TensorProto.FLOAT
tensor = onnx.helper.make_tensor(tensor_name, t_type, list(values.shape), values_raw, True)

# Write to file
f = open(filename + '.tensor', 'wb')
f.write(tensor.SerializeToString())
f.close()