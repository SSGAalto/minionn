"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import onnx
import onnx.numpy_helper

from . import config

import logging
logger = logging.getLogger('minionn.onnx_helper')

def stripModelFromPrivateData(m):
    """
    Strips the given model from all private data and returns a copy.
    This usually includes all tensors, i.e. all w and b.
    """
    # Create new model and copy all from m
    privatizedModel = onnx.ModelProto()
    privatizedModel.CopyFrom(m)

    # Clear the tensors from the model
    del privatizedModel.graph.initializer[:]

    # Return the privatized model
    return privatizedModel


def onnx_tensor_to_list(tensor):
    return onnx.numpy_helper.to_array(tensor).flatten().tolist()


def retrieveTensorsFromModel(model):
    tensor_dict = {}
    for t in model.graph.initializer:
        tensor_dict[str(t.name)] = onnx_tensor_to_list(t)
        logger.debug("Parsed tensor with name " + str(t.name) + ".")
        if config.debug_mode:
            logger.debug(" It starts with " + str(tensor_dict[str(t.name)][:2]))
        
    return tensor_dict


def retrieveTensorDimensionsFromModel(model):
    dims_dict = {}
    for i in model.graph.input:
        dimensions = []
        for d in i.type.tensor_type.shape.dim:
            dimensions.append(d.dim_value)
        dims_dict[str(i.name)] = dimensions
    logger.debug("Tensor dimensions are:" + str(dims_dict))

    return dims_dict


# ONNX attribute type map as defined in the onnx protobuf file
# Only the required attributes here
_attr_type_map = {
    1: "f",
    2: "i",
    3: "s",
    4: "t",
    7: "ints"
}


def retrieveNodesFromModel(model):
    nodes = model.graph.node
    nodes_list = []
    for n in nodes:
        node_dict = {"input":n.input,
                     "output":n.output,
                     "operation":n.op_type,
                     "attributes":[
                            {"name":a.name,
                             "type":a.type,
                             "value":getattr(a,_attr_type_map[a.type])}
                             for a in n.attribute]
         }
        nodes_list.append(node_dict)

    logger.debug("Nodes of the network are: " + str(nodes_list))
    return nodes_list


def get_bs_and_ws(nodes, tensors):
    b_list = []
    w_list = []

    # Parse the nodes for any occurence of Gemm and put the inputs into b and w accordingly
    for n in nodes:
        if n["operation"] == "Gemm":
            inp = n["input"]
            # Get the two operators
            # In a normal W*x matrix multiplication, w is the first input
            w = inp[0]
            x = inp[1]

            #Figure out which one is w
            if w not in tensors and x in tensors:
                # we have x*w 
                # minionn expects W*x
                # The difference is that for x*w we need to transpose w for 
                # the precomputation phase (U)
                # However, in this step this does not matter as we only 
                # use the Ws and Bs for writing into the model (see fractions)
                w = x
            else:
                logger.error("Retrieval of bs and ws from matrix multiplications failed! No w found:" + str(inp))

            # Store them to lists
            w_list.append(w)
            b_list.append(inp[2])

    return b_list, w_list


def calculate_input_dependent_tensors(nodes, model_input):
    input_dependent_tensors = [model_input]
    for n in nodes:
        input_Set = set(input_dependent_tensors)
        operator_Set = set(n["input"])
        set_intersection = input_Set & operator_Set

        if len(set_intersection) > 0:
            # Input of node depends on model input
            # Add output of node to the dependent set
            input_dependent_tensors.append(n["output"][0])
            
    return input_dependent_tensors