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
model = onnx.load("R2_S.onnx")

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
tensor_dict["8"] = np.reshape(tensor_dict["1"], (10,784))

# do fractionals
fractional = 1000
downscale = 1

single = ["2","4","6","8"]
double = ["3","5","7"]
for s in single:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional)

for s in double:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional*fractional)

"""
for s in tensor_dict:
    tensor_dict[s] = np.array([int(d) for d in tensor_dict[s].flatten().tolist()]).reshape(tensor_dict[s].shape)
"""

# compute
tensor_dict["11temp"] = np.matmul(tensor_dict["8"], tensor_dict["2"].T)
tensor_dict["11add"] = np.add(tensor_dict["11temp"], tensor_dict["3"])
tensor_dict["11"] = np.divide(tensor_dict["11add"],fractional*downscale)

tensor_dict["12"] = np.maximum(tensor_dict["11"],0)

tensor_dict["15temp"] = np.matmul(tensor_dict["12"], tensor_dict["4"].T)
tensor_dict["15add"] = np.add(tensor_dict["15temp"], tensor_dict["5"])
tensor_dict["15"] = np.divide(tensor_dict["15add"],fractional*downscale)

tensor_dict["16"] = np.maximum(tensor_dict["15"],0)

tensor_dict["19temp"] = np.matmul(tensor_dict["16"], tensor_dict["6"].T)
tensor_dict["19add"] = np.add(tensor_dict["19temp"], tensor_dict["7"])
tensor_dict["19"] = np.divide(tensor_dict["19add"],fractional*downscale)


given = np.loadtxt("out.txt", delimiter=",").astype(int)
diff = np.subtract(given, tensor_dict["19"])

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

print("expected (19)")
print(tensor_dict["19"])

print("given")
print(given)

print("Difference between expected and given result:")
print(diff)

print("\n")
print("Prediction expected: " + str(np.amax(tensor_dict["19"])) + " at index " + str(np.argmax(tensor_dict["19"])) )
print("Prediction have: " + str(np.amax(given))+ " at index " + str(np.argmax(given)))
print("\n")

#np.testing.assert_array_equal(tensor_dict["19"], given, err_msg="Result is not the same as expected result!", verbose=True)

if np.argmax(tensor_dict["19"]) == np.argmax(given):
    print("Prediction result equal. Test passed.")
else:
    print("Prediction differs. TEST FAILED!")