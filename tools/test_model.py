"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import onnx
from onnx import numpy_helper
import numpy as np

print("These tests work with a fractional of 1 and a downscale of 10,000. Set this in config.py accordingly before running MiniONN.")

model = onnx.load("manual_model.onnx")

tensor_dict = {}
for t in model.graph.initializer:
    tensor_dict[str(t.name)] = onnx.numpy_helper.to_array(t)


input_tensor = onnx.TensorProto()
with open('manual_model.onnx.tensor', 'rb') as fid:
    content = fid.read()
    input_tensor.ParseFromString(content)

tensor_dict["1"] = onnx.numpy_helper.to_array(input_tensor)

# do fractionals
fractional = 1
downscale = 10000

single = ["1", "2", "6"]
double = ["3", "7"]
for s in single:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional)

for s in double:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional*fractional)

for s in tensor_dict:
    tensor_dict[s] = np.array([int(d) for d in tensor_dict[s].flatten().tolist()]).reshape(tensor_dict[s].shape)

tensor_dict["4temp"] = np.matmul(tensor_dict["1"], tensor_dict["2"])
tensor_dict["4added"] = np.add(tensor_dict["4temp"], tensor_dict["3"])
tensor_dict["4"] = np.divide(tensor_dict["4added"],fractional*downscale).astype(int)

tensor_dict["5"] = np.maximum(tensor_dict["4"],0)

tensor_dict["8temp"] = np.matmul(tensor_dict["5"], tensor_dict["6"])
tensor_dict["8added"] = np.add(tensor_dict["8temp"], tensor_dict["7"])
tensor_dict["8"] = np.divide(np.maximum(tensor_dict["8added"],0),fractional*downscale).astype(int)

"""
print("W1")
print(tensor_dict["2"])

print("b1")
print(tensor_dict["3"])

print("Before Relu")
print(tensor_dict["4"])

print("After Relu")
print(tensor_dict["5"])

print("W2")
print(tensor_dict["6"])

print("b2")
print(tensor_dict["7"])
"""

print("Expected result")
print(tensor_dict["8"])

# now see if the result is close
given = np.loadtxt("out.txt", delimiter=",")
diff = np.subtract(given, tensor_dict["8"])

print("Given result")
print(given)

print("Diff")
print(diff)


np.testing.assert_array_equal(tensor_dict["8"], given, err_msg="Result is not the same as expected result!", verbose=True)

print("All numbers equal. Test passed")

