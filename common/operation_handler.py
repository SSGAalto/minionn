"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

# Logging
import logging
logger = logging.getLogger('minionn.NodeOperator')

# Wrapper for cpp functions and vectors 
from  . import minionn_helper

# Helper for onnx functions
from . import onnx_helper

# Node operations
from common.node_operations import log, gemm, relu, reshape, softmax

import copy
from operator import mul
from functools import reduce

class OperationHandler(object):
    """
    Class that takes the nodes of the model and transforms them into
     node objects that will handle the execution of that layer.
    Is called by client and server to execute their respective version of each layer. 
    """

    """
    Map of the operations named in the nodes of an ONNX file
     to their corresponding function
    Defined in: https://github.com/onnx/onnx/blob/master/docs/Operators.md

    Log and Softmax are not supported by MiniONN but are kept here as placeholders.
     This allows to run models with Softmax and Log in them without needing to change the ONNX model.
    """
    _ONNX_OPERATIONS = {
        "Log": log.LogNode, # Not supported and ignored
        "Gemm": gemm.GeneralMatrixMultiplicationNode,
        "Relu": relu.ReluNode,
        "Reshape": reshape.ReshapeNode,
        "Softmax": softmax.SoftmaxNode, # Not supported and ignored
    }

    def __init__(self, nodes, model_input, simulation=False):
        # Use a copy of the node dictionary to prevent side effects
        self._nodes = copy.deepcopy(nodes)

        # Now, calculate which tensors are depending on the input
        # This is necessary to know for e.g. Gemm where the detection
        #  of W and x (for W*x or x*W) depends on which tensors is
        #  depending on the input
        input_dependent_tensors = onnx_helper.calculate_input_dependent_tensors(self._nodes, model_input)

        # If we are in simulation mode, log more about the model
        if simulation:
            logger.info("Tensors that depend on the model input are:" 
                + str(input_dependent_tensors)
            )

        # Build the order of operations
        self._operations = []
        self.w_list = []
        gemm_count = 0
        for n in self._nodes:
            new_operation = self._get_node_operator(n, input_dependent_tensors)
            self._operations.append(new_operation)

            # Now depending on the operator:
            #  - For Gemm, store name of w for precomputation and store the index of this Gemm
            #  - For Relu, store the index of the previous Gemm (for r detection)
            if type(new_operation) is gemm.GeneralMatrixMultiplicationNode:
                self.w_list.append((new_operation.get_w(), new_operation.get_dimensions()))
                new_operation.set_gemm_count(gemm_count)
                gemm_count += 1

                logger.debug("Detected W as " + str(self.w_list[-1][0]) 
                + " with shape of gemm (m,n,o) " + str(self.w_list[-1][1]))

            if type(new_operation) is relu.ReluNode:
                new_operation.set_gemm_count(gemm_count)

        # Sanity check that all w's got detected
        # Technically, this should not happen and only happens
        #  when a weight matrix of a Gemm is modified before being used.
        # Then, the input to the Gemm is different than the initializer matrix and
        #  as such is not detected.
        # If you experience this error, go through your node operations and 
        #  make sure that you execute all nodes that change a weight matrix BEFORE
        #  the MiniONN precomputation (as this requires all W's to be present)
        assert len(self.w_list) == gemm_count, \
            "Not all w's were properly detected by the Matrix Multiplications!"

        # Print the order of operations of the network in simulation mode
        if simulation:
            logger.info("Network has this order:")
            for o in self._operations:
                logger.info(" - " + str(o))
            logger.info("Network end.")

    def init_server(self, instance_u):
        """
        Initializes the NodeOperator in server mode.
        """
        self._is_server = True
        self._instance_u = instance_u

        # Iterate over operations and:
        #  - for Gemm: fill in Us
        u_counter = 0
        for o in self._operations:
            if type(o) is gemm.GeneralMatrixMultiplicationNode:
                # get dimensions of o
                dims = o.get_dimensions()
                # fill in U of o
                this_u = minionn_helper.extract_sum(
                    self._instance_u, 
                    dims, 
                    u_counter
                )
                o.set_u(this_u)
                u_counter += reduce(mul, dims, 1)


    def init_client(self):
        """
        Initialize the NodeOperator in client mode.
        """
        self._is_server = False

        """
        Iterate through the operations and revert all changes on R that might happen
        between the matrix multiplication and its activation function / beginning.
        This is necessary as the r might change between its first usage and
         the Gemm
        The simplest example is a transpose in the Gemm. There, the r might
         be used as a transposed r but for the preceding activation 
         function it needs to not be transposed.
        The rs are generated as they are used in the Gemms, so we might need
         to change some r's here (e.g. transpose them)
        """

        mm_counter = 0
        for i in range(0,len(self._operations)):
            o = self._operations[i]
            if type(o) is gemm.GeneralMatrixMultiplicationNode:
                # This is a Gemm. Now adjust r based on each operation                
                #Go back to the beginning or to last consumer of r
                j = i
                current_r = "initial_r" + str(mm_counter)
                while j >= 0 and not self._operations[j].consumes_r(): 
                    # Call the operation to reverse its effect on r
                    current_r = self._operations[j].reverse_r(current_r)
                    
                    j -= 1
                
                # Now we have reversed all effects for the r that the activation function needs
                # store this in xr where x is the number of the Gemm operation
                r_name = "r" + str(mm_counter)
                minionn_helper.put_cpp_tensor(r_name, 
                    minionn_helper.get_cpp_tensor(current_r),
                    minionn_helper.get_cpp_tensor_dim(current_r))

                logger.debug("Rewrote initial_r" + str(mm_counter) 
                    + " to " + current_r 
                    + " and stored it in " + r_name
                    + ". It's shape is now " 
                    + str(minionn_helper.get_cpp_tensor_dim(r_name))
                )

                # increment matrix multiplication counter
                mm_counter += 1

    def get_w_list(self):
        return self.w_list

    def run_network(self, x_in, in_name, out_name):
        logger.info("RUNNING NETWORK")
        logger.debug("Desired output:" + out_name)

        # Put x into the tensor dictionary
        # it should already be in there as a dimension stub,
        #  so take the dimension from there
        x_dim = minionn_helper.get_cpp_tensor_dim(in_name)
        minionn_helper.put_cpp_tensor(in_name, x_in, x_dim)

        # Execute
        for o in self._operations:
            self._execute_operator(o)
            minionn_helper._log_vector_dict()
    
        #We are done if we have the output
        if minionn_helper.has_cpp_tensor(out_name) and \
            minionn_helper.tensor_has_values(out_name):
            # Return requested output
            return minionn_helper.get_cpp_tensor(out_name)

        # If we did not compute the output until here, we have a problem
        logger.error("Requested output was not calculated: " + str(out_name))

    def _get_node_operator(self, node, input_dependent_tensors):
        """
        Returns a new node that is registered as responsible for the given ONNX operator.
        """
        op_type = node["operation"]
        if op_type in self._ONNX_OPERATIONS:
            return self._ONNX_OPERATIONS[op_type](node, input_dependent_tensors)
        else:
            raise TypeError("ONNX node of type {} is not supported.".format(op_type))

    def _execute_operator(self, operator):
        """
        Run a given node operator
        The executed function depends if we are server or client
        """
        if self._is_server:
            operator.server()
        else:
            operator.client()