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

model = onnx.load("manual_model_only_gemm.onnx")

tensor_dict = {}
for t in model.graph.initializer:
    tensor_dict[str(t.name)] = onnx.numpy_helper.to_array(t)


input_tensor = onnx.TensorProto()
with open('manual_model_only_gemm.onnx.tensor', 'rb') as fid:
    content = fid.read()
    input_tensor.ParseFromString(content)

tensor_dict["1"] = onnx.numpy_helper.to_array(input_tensor)
#tensor_dict["1"] = np.reshape(tensor_dict["1temp"], (1,3))


# do fractionals
fractional = 1
downscale = 10000

single = ["1", "2"]
double = ["3"]
for s in single:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional)

for s in double:
    tensor_dict[s] = np.multiply(tensor_dict[s], fractional*fractional)

for s in tensor_dict:
    tensor_dict[s] = np.array([int(d) for d in tensor_dict[s].flatten().tolist()]).reshape(tensor_dict[s].shape)

tensor_dict["4temp"] = np.matmul(tensor_dict["2"], tensor_dict["1"])
tensor_dict["4added"] = np.add(tensor_dict["4temp"], tensor_dict["3"])
tensor_dict["4"] = np.divide(tensor_dict["4added"],fractional*downscale).astype(int)

"""
print("Input")
print(tensor_dict["1"])

print("W1")
print(tensor_dict["2"])

print("b1")
print(tensor_dict["3"])
"""

print("Expected result")
print(tensor_dict["4"])

# now see if the result is close
given = np.loadtxt("out.txt", delimiter=",").astype(int)
diff = np.subtract(given, tensor_dict["4"])

print("Given result")
print(given)

print("Diff")
print(diff)


np.testing.assert_array_equal(tensor_dict["4"], given, err_msg="Result is not the same as expected result!", verbose=True)

print("All numbers equal. Test passed")