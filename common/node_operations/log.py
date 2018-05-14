"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
Placeholder operation for Log. Does nothing.

Log is currently not supported by MiniONN and is ignored!

"""
from .common import BaseNode
from common import minionn_helper
import logging
logger = logging.getLogger('minionn.node_operations')

class LogNode(BaseNode):
    def __init__(self, node, input_dependent_tensors):
        super(LogNode, self).__init__(node, input_dependent_tensors)

        # Input is a single vector (and not several)
        self.inp = self.inp[0]

        # Store the output tensor with original dimensions (they dont change)
        dims = minionn_helper.get_cpp_tensor_dim(self.inp)
        minionn_helper.put_cpp_tensor(
            self.outp, 
            None,
            dims
        )

        logger.debug("Operation Log will make " 
            + minionn_helper.print_tensor(self.inp)
            + " into tensor " + self.outp + " of shape "
            + str(dims)
        )

        #For to string, store name
        self._string_name = "Log"
    
    def server(self):
        """
        Placeholder function to simulate execution of a node
        """
        logger.info("Log operation. Redirected input to output. Did nothing.")
        logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
        logger.debug("We have the attributes " + str(self.node["attributes"]))
        logger.debug("Log end")

        # Copy input tensor over to output
        minionn_helper.copy_tensor(self.inp, self.outp)
    
    def client(self):
        """
        Placeholder function to simulate execution of a node
        """
        if minionn_helper.tensor_has_values(self.inp):
            # If this tensor exists, simply reshape it.
            self.server()
        else:
            logger.info("Log operation did nothing.")
            logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
            logger.debug("We have the attributes " + str(self.node["attributes"]))
            logger.debug("Log end")

        # If this tensor does not exist here, do nothing
        #  We already stored a stub tensor during initialization