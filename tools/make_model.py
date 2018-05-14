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

"""
 Define the shape of the input
 Also define the operators with their shapes.
 Example:
  ["Gemm", (5,10,20), ["1","2","3"], ["4"]]
  means a Gemm onnx operator with 1*2+3=4
   with the shapes
       1 : 5,10
       2 : 10,20
       3 : 5
       4 : 5, 20
 Operators see https://github.com/onnx/onnx/blob/master/docs/Operators.md

 This script always assumes that the input is multiplied first: x * W + b
 This can obviously be changed but the auto generation below assumes W to be 
  at second place etc.
"""
model_name = "manual_model"
operators = [ 
    ["Gemm", (2,3,4), ["1", "2", "3"], ["4"]], 
    ["Relu", (2,4), ["4"], ["5"]],  
    ["Gemm", (2,4,2), ["5", "6", "7"], ["8"]] 
]


""" 
    Helper functions
"""
def generate_random_tensor(name, shape):
    # Create random numpy input
    x = np.random.randint(10000,None,size=shape,dtype='int64')
#    x = np.array([[random.randint(0,10000) for n in range(0,cols) ] for m in range(0,rows)])
    x_l = x.flatten().tolist()

    # Pack the input into raw bytes
    x_raw = struct.pack('%sf' % len(x_l), *x_l)

    # export the raw data to a tensor proto
    t_type = onnx.TensorProto.FLOAT
    t = onnx.helper.make_tensor(name, t_type, list(x.shape), x_raw, True)

    return t

"""
 First, create the input
"""
x_t = generate_random_tensor(operators[0][2][0], (operators[0][1][0], operators[0][1][1]))

# Write to file
f = open(model_name + '.onnx.tensor', 'wb')
f.write(x_t.SerializeToString())
f.close()

"""
 Second, create the nodes
 Here, we have Gemm, Relu, Gemm
"""
nodes = []

for o in operators:
    name = o[0]
    inputs = o[2]
    outputs = o[3]
    nodes.append(onnx.helper.make_node(name, inputs, outputs))

"""
 Next, initializers (List of initial tensors)
 Create a list of tensors. Helpful for this is the shape tuple in operators.
"""
initializers = []
initializers_value_info = []
for o in operators:
    if o[0] == "Gemm":
        # We need two randoms, W and b
        w_name = o[2][1]
        b_name = o[2][2]
        w_shape = (o[1][1], o[1][2])
        b_shape = [o[1][2]]
        initializers.append(generate_random_tensor(w_name, w_shape))
        initializers.append(generate_random_tensor(b_name, b_shape))

        initializers_value_info.append(
            onnx.helper.make_tensor_value_info(
                w_name, onnx.TensorProto.FLOAT,w_shape
        ))
        initializers_value_info.append(
            onnx.helper.make_tensor_value_info(
                b_name,onnx.TensorProto.FLOAT,b_shape
        ))

"""
 Lastly, graph and model
"""
graph_inputs = [
    onnx.helper.make_tensor_value_info(
        operators[0][2][0], 
        onnx.TensorProto.FLOAT, 
        (operators[0][1][0],operators[0][1][1])
    )
]
graph_inputs.extend(initializers_value_info)


graph_outputs = [
    onnx.helper.make_tensor_value_info(
        operators[-1][3][0], 
        onnx.TensorProto.FLOAT, 
        (operators[-1][1][0],operators[-1][1][-1])
    )
]


graph = onnx.helper.make_graph(nodes, name, graph_inputs, graph_outputs, initializer=initializers)
model = onnx.helper.make_model(graph)

# Write to file
f = open(model_name + '.onnx', 'wb')
f.write(model.SerializeToString())
f.close()