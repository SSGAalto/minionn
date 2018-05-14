"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

"""
Operator node for Gemm.
Adhering to https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm

Gemm

General Matrix multiplication: 
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3 
Compute Y = alpha * A * B + beta * C, where input tensor A has 
dimension (M X K) , input tensor B has dimension (K X N), 
input tensor C and output tensor Y have dimension (M X N). 
If attribute broadcast is non-zero, input tensor C will be broadcasted 
to match the dimension requirement. A will be transposed before 
doing the computation if attribute transA is non-zero, same for B and transB.
Version
This version of the operator has been available since version 6 of the 
default ONNX operator set.

Other versions of this operator: Gemm-1
Attributes

alpha : float
    Scalar multiplier for the product of input tensors A * B
beta : float
    Scalar multiplier for input tensor C
broadcast : int
    Whether C should be broadcasted
transA : int
    Whether A should be transposed
transB : int
    Whether B should be transposed

Inputs

A : T
    Input tensor A
B : T
    Input tensor B
C : T
    Input tensor C, can be inplace.

Outputs

Y : T
    Output tensor.

Type Constraints

T : tensor(float16), tensor(float), tensor(double)
    Constrain input and output types to float tensors. 


NOTE: Operator currently only supports transA and transB. Alpha, Beta, Broadcast are ignored.
"""
from .common import BaseNode
from common import minionn_helper
from common import config
import logging
logger = logging.getLogger('minionn.node_operations')

class GeneralMatrixMultiplicationNode(BaseNode):
    def __init__(self, node, input_dependent_tensors):
        super(GeneralMatrixMultiplicationNode, self).__init__(node, input_dependent_tensors)

        # Get dimensions for this Gemm
        self.dim_w = minionn_helper.get_cpp_tensor_dim(self.inp[0])
        self.dim_x = minionn_helper.get_cpp_tensor_dim(self.inp[1])

        self.dim_m =   self.dim_w[0]
        self.dim_n =   self.dim_w[1]
        self.dim_n_c = self.dim_x[0]
        self.dim_o =   self.dim_x[1]

        self._broadcast_active = False

        logger.debug("Gemm start. Dimensions are " + str(self.get_dimensions()) + " with inputs " + str(self.inp))

        # Analyze and parse attributes
        attribute_map = {
            "alpha": self._alpha,
            "beta": self._beta,
            "transA": self._transA,
            "transB": self._transB,
            "broadcast": self._broadcast
        }
        for a in self.node["attributes"]:
            attribute_map[a["name"]](a)

            logger.debug("Gemm attribute " + a["name"]  + ". Dimensions are " + str(self.get_dimensions()) + " with inputs " + str(self.inp))

        # Sanity check if dimensions match
        if(self.dim_n != self.dim_n_c):
            logger.error(
                "Matrix multiplication: Dimensions do not match! "
                + "They are: [" + str(self.dim_m) + "," + str(self.dim_n) + "] "
                + "[" + str(self.dim_n_c) + "," + str(self.dim_o) + "] "
        )

        # If broadcast is Not active, throw a warning
        if not self._broadcast_active:
            logger.error("Broadcast is not active but we are broadcasting anyways! If you do not want broadcast, check the Matrix Multiplication functions.")
            
        # Already create the output stub with correct dimensions
        # The actual output is not affected by the following potential
        #  dimension adjustments
        minionn_helper.put_cpp_tensor(
            self.outp, 
            None,
            [self.dim_m, self.dim_o]
        )

        # Minionn needs w * x and not x * w !
        # If we have W*x, calculate W*x + U + b
        # If we have x*w, instead calculate (W' * X' + U + b')' 
        #   (U is already calculated accordingly and needs no transpose here)
        if self.inp[1] in self.input_dependent_tensors:
            # First input is a W. We have W*x. Proceed normally
            logger.debug("Normal W*x + b + U Gemm")
            self.order_w_x = True
            self.w = self.inp[0]

        elif self.inp[0] in self.input_dependent_tensors:
            # First input is not a W, we have x*W. Perform alternative multiplication
            # calculate (W' * X' + U)' + b . In one step: (W' * X' + U + b')'
            logger.debug("Reversed Gemm with x*w + b. Computing (W'*x' + U + b')' instead")
            new_x = self.inp[0] + "T"
            new_w = self.inp[1] + "T"
            new_b = self.inp[2]
            self.inp = [new_w, new_x, new_b ]
            self.order_w_x = False
            self.w = new_w

            # Adjust dimensions
            # We had m x n * n x o and now have o x n * n x m
            # --> Swap m and o
            self.dim_m, self.dim_o = self.dim_o, self.dim_m
        else:
            logger.error("Both inputs to matrix multiplication depend on the model input. This is not supported by MiniONN as we cannot perform a precomputation then.")
        
        logger.debug("Operation Gemm will operate on " 
            + str(self.inp) + " to " + self.outp + " with the following dimensions "
            + str(self.dim_m) + "x" + str(self.dim_n)
            + " * " + str(self.dim_n_c) + "x" + str(self.dim_o)
        )

        #For to string, store name
        self._string_name = "Gemm"

    def set_u(self, u):
        self.u = u

    def set_gemm_count(self, count):
        self.gemm_count = count

    def get_w(self):
        return self.w

    def get_dimensions(self):
        """
        Returns the __actual__ dimensions of this Gemm.
        This accounts for:
         - TransA, TransB
         - Wx or xW situation
        """
        return [self.dim_m, self.dim_n, self.dim_o]

    def has_order_w_x(self):
        return self.order_w_x
    
    def server(self):
        logger.info("Performing Gemm with " 
                + str(self.inp) + " to " + self.outp + " with the following dimensions "
                + str(self.dim_m) + "x" + str(self.dim_n)
                + " * " + str(self.dim_n_c) + "x" + str(self.dim_o)
        )

        minionn_helper.matrix_mult(self.inp, self.outp, self.u, order_w_x=self.order_w_x)


    def client(self):
        logger.info("Performing Client Gemm with " + str(self.inp) + " to " + self.outp )

        minionn_helper.matrix_mult_client(self.inp, self.outp, "v" + str(self.gemm_count), order_w_x=self.order_w_x)
    
    def reverse_r(self, current_r):
        """
        A reversed matrix multiplication means that due to MiniONN, we transpose
         both inputs and reverse their order (as MiniONN expects W*x)
        For r, this means that we need to transpose it for its earlier use 
         in the activation function/client start.
        """
        if self.order_w_x:
            logger.debug("Gemm reverse r: NOT reversing r. Returning " + current_r)
            return current_r
        else:
            logger.debug("Gemm reverse r: REVERSING r. Returning " + current_r +"T")
            return current_r + "T"
    
    def _alpha(self, attribute):
        # Throw error if alpha value is not 1 (not supported currently)
        if attribute["value"] != 1.0:
            logger.error("Gemm attribute alpha is not 1. This is unsupported. Ignoring alpha.")
    
    def _beta(self, attribute):
        # Throw error if beta value is not 1 (not supported currently)
        if attribute["value"] != 1.0:
            logger.error("Gemm attribute beta is not 1. This is unsupported. Ignoring beta.")

    def _transA(self, attribute):
        if attribute["value"] == 1:
            # Transpose A (Deferred transpose 
            #  -> Transpose is executed when tensor is accessed)
            # Store the transposed vector as input and swap dimensions
            self.inp[0] = self.inp[0] + "T"
            self.dim_m, self.dim_n = self.dim_n, self.dim_m
    
    def _transB(self, attribute):
        if attribute["value"] == 1:
            # Transpose B (Deferred transpose 
            #  -> Transpose is executed when tensor is accessed)
            # Store the transposed vector as input and swap dimensions
            self.inp[1] = self.inp[1] + "T"
            self.dim_o, self.dim_n_c = self.dim_n_c, self.dim_o
    
    def _broadcast(self, attribute):
        # Currently, broadcasting is always on (we assume a column vector and add it to the matrix)
        # If you wish to implement this properly, change b here and change the matrix multiplication code that is called
        if attribute["value"] == 1:
            self._broadcast_active = True
            

        