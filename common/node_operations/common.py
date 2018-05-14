"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
Placeholder operation. Does nothing.

Is used as a base class for the other operations.

Operations should be adhere to the standards defined in
https://github.com/onnx/onnx/blob/master/docs/Operators.md

"""
import logging
logger = logging.getLogger('minionn.node_operations')

class BaseNode(object):
    def __init__(self, node, input_dependent_tensors):
        # Store node and inputs/output
        self.node = node
        self.inp = node["input"]
        self.outp = node["output"][0]

        # We only care about the inputs to our node
        self.input_dependent_tensors = list(set(input_dependent_tensors) & set(self.inp))

        # Store if this operation consumes an r in the MiniONN context
        # The idea behind this is that we need to know which operations need an r
        #  to function. In general these are the activation functions as those
        #  use multi party computations and the r for the next matrix multiplication
        # As such, a "Gemm" operation does NOT __consume__ an r. Instead, the r is
        #  consumed before by either the input into the Gemm or by its preceding
        #  activation function such as Relu.
        self._consumes_r = False

        #For to string, store name
        self._string_name = "Base Node Operation"

    
    def client(self):
        """
        NOTE: Implement this function for the client operation.
        Computes the Node on the client.
        """
        logger.info("Placeholder for Client operation.")
        logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
        logger.debug("We have the attributes " + str(self.node["attributes"]))
        logger.debug("Placeholder end")
    
    def server(self):
        """
        NOTE: Implement this function for the server operation.
        Computes the Node on the server.
        """
        logger.info("Placeholder for Server operation.")
        logger.debug("Inputs are " + str(self.node["input"]) + " while output is into " + str(self.node["output"]))
        logger.debug("We have the attributes " + str(self.node["attributes"]))
        logger.debug("Placeholder end")

    def reverse_r(self, current_r):
        """
        NOTE: Implement this function for the client.
        Reverses the effects this node has on MiniONN's r.
        Such effects might not be trivial and have to be handled with care.
        The problem with r is that it is random and has the shape of x
          AT THE TIME of the matrix multiplication.
        This means if the matrix multiplication actually computes on x', 
         r needs to be transposed too (to r' then) for its usage in the 
         activation function so that X_s is correct shape when it
         comes to the matrix multiplication (where r is assumed).
        
        See Gemm operator for an example of this.

        The default implementation does nothing and simply returns 
         the r without changing it.
        """
        return current_r


    def consumes_r(self):
        return self._consumes_r

    def __str__(self):
     return self._string_name + ". Inputs are " + str(self.inp) + " while output is into " + str(self.outp)
