"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
Operator node for Reshape.
Adhering to https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape


Reshape

Reshape the input tensor similar to numpy.reshape.

First input is the data tensor, second input is a shape tensor which 
specifies the output shape. It outputs the reshaped tensor.

At most one dimension of the new shape can be -1. In this case, 
the value is inferred from the size of the tensor and the 
remaining dimensions. A dimension could also be 0, in which case 
the actual dimension value is unchanged (i.e. taken from the input tensor).

Version
This version of the operator has been available since version 5 of 
the default ONNX operator set.

Other versions of this operator: Reshape-1
Inputs

data : T
    An input tensor.
shape : tensor(int64)
    Specified shape for output.

Outputs

reshaped : T
    Reshaped data.

Type Constraints

T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors.

"""
from .common import BaseNode
import logging
logger = logging.getLogger('minionn.node_operations')
from common import minionn_helper

from operator import mul
from functools import reduce

import numpy as np

class ReshapeNode(BaseNode):
    def __init__(self, node, input_dependent_tensors):
        super(ReshapeNode, self).__init__(node, input_dependent_tensors)
        self.inp = self.inp[0]

        # First, calculate dimensions
        self.dims = list(self.node["attributes"][0]["value"])
        # if it contains a -1, calculate that entry by deduction
        if self.dims.count(-1) > 1:
            logger.error("Reshape only works with at most one -1 dimension.")

        if -1 in self.dims:
            i = self.dims.index(-1)
            # Calculate this dimension
            total_size = reduce(mul, minionn_helper.get_cpp_tensor_dim(self.inp), 1)
            other_dims = [d for d in self.dims if d != -1]
            current_size = reduce(mul, other_dims, 1)
            self.dims[i] = int(total_size / current_size)

        # And store that tensor with the calculated dimensions
        minionn_helper.put_cpp_tensor(
            self.outp, 
            None,
            self.dims
        )

        logger.debug("Operation Reshape will reshape " 
            + minionn_helper.print_tensor(self.inp)
            + " into tensor " + self.outp + " of shape "
            + str(self.dims)
        )

        #For to string, store name
        self._string_name = "Reshape"

    def server(self):
        # Use numpy to reshape input
        # Technically, we use cpp vectors here and would not need
        #  to touch the vector at all. But just to avoid any problems
        #  once we change away from cpp vectors, lets use numpy and copy here.
        original = minionn_helper.get_cpp_tensor(self.inp)
        reshaped = np.reshape(
                original,
                self.node["attributes"][0]["value"]
            )
        dims = list(reshaped.shape)

        logger.info("Reshaping tensor " 
            + minionn_helper.print_tensor(self.inp)
            + " into tensor " + self.outp + " of shape "
            + str(dims) 
            + ". Original length is " + str(len(original)) 
            + " and resized has size " + str(len(reshaped))
        )
        
        minionn_helper.put_cpp_tensor(
            self.outp, 
            reshaped.flatten().tolist(),
            dims
        )

    def client(self):
        if minionn_helper.tensor_has_values(self.inp):
            # If this tensor exists, simply reshape it.
            self.server()
        else:
            logger.info("Client Reshape did nothing." 
                + minionn_helper.print_tensor(self.inp)
                + " (I do not have this one) should be reshaped to tensor " + self.outp + " of shape "
                + str(self.dims)
            )

        # If this tensor does not exist here, do nothing
        #  We already stored a stub tensor during initialization
    
    