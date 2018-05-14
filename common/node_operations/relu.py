"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
Relu operation
https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu

Relu takes one input data (Tensor) and produces one output data (Tensor) 
where the rectified linear function, y = max(0, x), is applied to 
the tensor elementwise.

Version
This version of the operator has been available since version 6 of 
the default ONNX operator set.

Other versions of this operator: Relu-1
Inputs

X : T
    Input tensor

Outputs

Y : T
    Output tensor

Type Constraints

T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors. 
"""
from .common import BaseNode
from common import minionn_helper
import logging
logger = logging.getLogger('minionn.node_operations')

class ReluNode(BaseNode):
    def __init__(self, node, input_dependent_tensors):
        super(ReluNode, self).__init__(node, input_dependent_tensors)

        # Relu is an activation function that requires an R
        self._consumes_r = True

        # Input is a single tensor (and not several)
        self.inp = self.inp[0]

        # Store the output tensor with original dimensions (they dont change)
        dims = minionn_helper.get_cpp_tensor_dim(self.inp)
        minionn_helper.put_cpp_tensor(
            self.outp, 
            None,
            dims
        )

        logger.debug("Operation Relu will make " 
            + minionn_helper.print_tensor(self.inp)
            + " into tensor " + self.outp + " of shape "
            + str(dims)
        )

        #For to_string, store name
        self._string_name = "Relu"

    def set_gemm_count(self, count):
        self.gemm_count = count
    
    def server(self):
        """
        Relu server simply calls the cpp MPC code
        """
        logger.info("Relu operation. Server version.")
        logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
        logger.debug("We have the attributes " + str(self.node["attributes"]))

        minionn_helper.relu_server(self.inp, self.outp)
        logger.debug("Relu end")
    
    def client(self):
        """
        Relu client uses the R of the next layer during the MPC
        """
        logger.info("Relu operation. Client version.")
        logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
        logger.debug("We have the attributes " + str(self.node["attributes"]))
        
        rc = "r" + str(self.gemm_count)
        logger.debug("R is " + rc)
        minionn_helper.relu_client(self.inp, self.outp, rc)
        logger.debug("Relu end")