"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import onnx
import struct
import random
import numpy as np
import onnx
from onnx import numpy_helper
model = onnx.load("R_S.onnx")

def build(inp, shape):
    return np.array(inp).reshape(shape)

tensor_dict = {}
for t in model.graph.initializer:
    tensor_dict[str(t.name)] = onnx.numpy_helper.to_array(t)


input_tensor = onnx.TensorProto()
with open('S.tensor', 'rb') as fid:
    content = fid.read()
    input_tensor.ParseFromString(content)

tensor_dict["1"] = onnx.numpy_helper.to_array(input_tensor)
tensor_dict["6"] = np.reshape(tensor_dict["1"], (10,784))

# do fractionals
fractional = 1000
downscale = 1

single = ["2","4","6"]
double = ["3","5"]
for s in single:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional)

for s in double:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional*fractional)

"""
for s in tensor_dict:
    tensor_dict[s] = np.array([int(d) for d in tensor_dict[s].flatten().tolist()]).reshape(tensor_dict[s].shape)
"""

# compute
tensor_dict["9temp"] = np.matmul(tensor_dict["6"], tensor_dict["2"].T)
tensor_dict["9add"] = np.add(tensor_dict["9temp"], tensor_dict["3"])
tensor_dict["9"] = np.divide(tensor_dict["9add"],fractional*downscale)

tensor_dict["10"] = np.maximum(tensor_dict["9"],0)

tensor_dict["13temp"] = np.matmul(tensor_dict["10"], tensor_dict["4"].T)
tensor_dict["13add"] = np.add(tensor_dict["13temp"], tensor_dict["5"])
tensor_dict["13"] = np.divide(tensor_dict["13add"],fractional*downscale)

given = np.loadtxt("out.txt", delimiter=",").astype(int)
diff = np.subtract(given, tensor_dict["13"])

"""
print("Reshaped Input (6)")
print(tensor_dict["6"])

print("W1 (2)")
print(tensor_dict["2"])

print("b1 (3)")
print(tensor_dict["3"])

print("Before Relu (9)")
print(tensor_dict["9"])

print("After Relu (10)")
print(tensor_dict["10"])

print("W2 (4)")
print(tensor_dict["4"])

print("b2 (5)")
print(tensor_dict["5"])
"""

print("expected (13)")
print(tensor_dict["13"])
print("have")
print(given)

print("Difference between expected and given result:")
print(diff)

print("\n")
print("Prediction expected: " + str(np.amax(tensor_dict["13"])) + " at index " + str(np.argmax(tensor_dict["13"])) )
print("Prediction have: " + str(np.amax(given))+ " at index " + str(np.argmax(given)))
print("\n")

#np.testing.assert_array_equal(tensor_dict["13"], given, err_msg="Result is not the same as expected result!", verbose=True)

if np.argmax(tensor_dict["13"]) == np.argmax(given):
    print("Prediction result equal. Test passed.")
else:
    print("Prediction differs. TEST FAILED!")