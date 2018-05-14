"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

# cpp
import cppimport
import cppimport.import_hook
cppimport.set_quiet(True)
from lib import minionn as minionn

import logging
logger = logging.getLogger('minionn.minionn_helper')

from operator import mul
from functools import reduce

import numpy as np

from . import config

#TODO: Work with tensor objects held by a tensorhandler
class Tensor(object):
    """
    Class for a Tensor that can have a numpy or a cpp reference.
    Contains:
     - Shape
     - CPP representation (optional)
     - Numpy representation (optional but preferred)
    
    When the system gets initialized, all tensors are numpy tensors.
    Whenever a cpp tensor gets requested, it gets generated on demand from the numpy tensor.
    Vice versa, if a numpy tensor is requested but only a cpp vector exists, it also gets generated on demand (e.g. after a MPC operation, only a cpp vector will exist)

    During the computation of the network, only cpp tensors should be used. The main reason for this is that the MPC operations are implemented in cpp with the ABY library and require the tensors to be cpp vectors. Every conversion from cpp to numpy or vice versa contains at least one copy of the whole tensor which should be avoided whenever possible.
    """
    def __init__(self):
        pass

# NOTE: Currently NOT concurrency safe! Only works for a single client
# TODO: Refactor into tensors as objects. Then keep a dict of objects. 
# TODO: Think about splitting minionn helper and matrix stuff
# Maybe node operator can take care of matrices while having 
#  minionn helper only call cpp code
# Or create extra class for matrices and give it to node operator
#  as argument when calling run network

# TODO: Move minionn operations into the actual nodes

# dictionary to hold the cpp tensors mapped by their name
# This is a dict of a tensor name to a tuple of (VectorInt, Shape (as list))
cpp_tensors = {}

def _tensor_is_transposed(name):
    transposed = name.count("T") % 2
    if transposed:
        return True
    else:
        return False

def _tensor_get_base_name(name):
    return name.replace("T","")

def _tensor_normalize_name(name):
    normalized_name = _tensor_get_base_name(name)

    if _tensor_is_transposed(name):
        normalized_name += "T"
    
    return normalized_name

def _has_tensor(name):
    if name in cpp_tensors:
        return True
    return False

def _get_tensor(name):
    name_normalized = _tensor_normalize_name(name)

    if not _has_tensor(name_normalized) \
        or \
        cpp_tensors[name_normalized][0] is None:
        # Return transposed vector.
        # It does not exist yet, but we can quickly create it
        #  if its transposed exists and has values
        transposed_name = _tensor_normalize_name(name_normalized + "T")
        if _has_tensor(transposed_name) \
        and tensor_has_values(transposed_name):
            _transpose(transposed_name)

    return cpp_tensors[name_normalized][0]

def _get_tensor_dim(name):
    """
    Returns the shape (dimension) as a list of the given tensor.
    If name does not exist, we try to return the reversed dimension of 
    its transposed vector
    """
    tname = _tensor_normalize_name(name)
    if _has_tensor(tname):
        return cpp_tensors[tname][1]
    else:
        # Name does not exist. Try transposed of name
        tname = _tensor_normalize_name(name + "T")
        if _has_tensor(tname):
            return list(reversed(cpp_tensors[tname][1]))
        else:
            logger.error("Cannot get dimension of nonexistent tensor " + name)

def _set_tensor(name, new_tensor, new_dimension):
    cpp_tensors[_tensor_normalize_name(name)] = (new_tensor, new_dimension)
    logger.debug("Touched tensor " + name + ". New dims:" + str(new_dimension) )
    if new_tensor is not None and config.debug_mode:
        logger.debug("-- Tensor's size is " + str(len(list(new_tensor))))
#        assert + " size:" + str(new_tensor.size())

def _transpose(inp):
    """
    Takes the input vector from cpp_vectors, reshapes it into 
    the dimensions given, transposes the matrix, and creates a new,
    flattened, cpp vector as "<inp>T" with <inp> being the input string.
    """
    # Calculate new name (to prevent double T namings)
    new_name = _tensor_get_base_name(inp)
    if not _tensor_is_transposed(inp):
        new_name += "T"
    logger.debug("Transposing " + inp + " to output " + new_name )

    # Get vec and dim
    vec_in = list(cpp_tensors[inp][0])
    dim_in = cpp_tensors[inp][1]

    # Transpose the reshaped matrix
    reshaped = np.reshape(vec_in, dim_in)
    transposed = np.transpose(reshaped)
    dim_out = list(transposed.shape)

    # Flatten and store
    _set_tensor(new_name, minionn.VectorInt(transposed.flatten().tolist()), dim_out)

def put_cpp_tensor(name, values, dimension, fractional = 1):
    if values is None:
        # If values is none, input none to dict
        _set_tensor(name, None, dimension)
    elif fractional != 1 or not all(isinstance(v, int) for v in values):
        # If fractional is not 1 or we have a list of not solely integers, 
        # perform list comprehension
        tmp = [modulo_pmax(int(fractional * v)) for v in values]    
        _set_tensor(name, minionn.VectorInt(tmp), dimension)
    else: 
        # Else, simply add to dict
        _set_tensor(name, minionn.VectorInt(values), dimension)
    
def get_cpp_tensor(name, reshape = False):
    """
    Returns the cpp tensor associated with name.
    If name ends on T, the transposed tensor is returned.
    If reshape is true, a proper reshaped numpy array is returned
    """
    name_normalized = _tensor_normalize_name(name)

    tensor = list(_get_tensor(name_normalized))

    if reshape:
        # Use numpy to reshape array
        tensor = np.reshape(tensor, _get_tensor_dim(name_normalized))

    return tensor

def get_cpp_tensor_dim(name):
    """
    Returns the shape (dimension) as a list of the given tensor.
    Result is a list
    """
    return _get_tensor_dim(name)
    
def has_cpp_tensor(name):
    """
    Checks if the given named tensor exists.
    Takes the following three cases into account:
     - named vector exists
     - normal vector exists but transposed doesn't ("<name>T")
     - transposed vector "<name>T" exists but named vector doesn't
    """
    if  _has_tensor(_tensor_normalize_name(name)) \
        or _has_tensor(_tensor_get_base_name(name) + "T") \
        or _has_tensor(_tensor_get_base_name(name)):
        return True
    else:
        return False

def tensor_has_values(name):
    """
    Checks if a given tensor, if it exists, has any values or 
    if it just a stub for dimensions.
    """
    if has_cpp_tensor(name) and _get_tensor(name) is not None:
        return True
    else:
        return False

def print_tensor(name):
    normalized = _tensor_normalize_name(name)
    s = normalized
    if normalized != name:
        s += " (aka " + name + ")"
    s += " (dim: " + str(_get_tensor_dim(name)) + ")"
    
    if config.debug_mode:
        s += " (Currently has values: "  
        if tensor_has_values(name):
            s += "Yes. Complete size: " + str(len(get_cpp_tensor(name)))
            s += " First values:" + str(get_cpp_tensor(name)[:config.debug_print_length])
        else:
            s+= "No"

        s += ")"

    return s

def _log_vector_dict():
    logger.debug("Cpp dictionary elements:")
    
    for name in sorted(cpp_tensors):
        logger.debug("  -- " + print_tensor(name))
    logger.debug("End Cpp dictionary")

def copy_tensor(name_src, name_dst):
    src = _tensor_normalize_name(name_src)
    if _has_tensor(src):
        _set_tensor(name_dst, _get_tensor(src), _get_tensor_dim(src) )


""""
MiniONN functions.
These functions are CPP functions and the python functions are just a wrapper
for them.
"""
def init(slots):
    minionn.init(slots)

def init_mpc(ip, port, is_server):
    minionn.init_aby(ip, port, is_server)

def shutdown_mpc():
    minionn.shutdown_aby()

def generate_keys(pkey, skey):
    minionn.gen_keys(pkey, skey)

def server_prepare_w(w_list, pkey):
    """
    Prepares the W to send over to the client.
    This W contains all w from every matrix multiplication
     and is encrypted with the server's public key.
    Arranging the Ws is done doing the following:
     For each m x n * n x o matrix multiplication,
      this multiplication's W has every row of w repeated o times.
     Each multiplication's W is then attached to the overall W.

     Input: 
      - w_list: List of tuples:(name of W, dimensions of matrix multiplication [m,n,o])
      - public key of server
    """

    # We will use numpy to properly arrange the Ws.
    # In future, this has a way better performance if numpy is
    #  the primary library in use
    overall_w = []
    for (w, dim) in w_list:
        # Get list as reshaped numpy array
        tensor = get_cpp_tensor(w, reshape=True)

        for dm in range(0, dim[0]):
            for do in range(0, dim[2]):
                overall_w.extend(tensor[dm].tolist())

    if config.debug_mode:
        logger.debug("W has size " + str(len(overall_w)))
        logger.debug("W starts with " + str(overall_w[:config.debug_print_length_long]) + " and ends with " + str(overall_w[-config.debug_print_length_long:]))

    return minionn.encrypt_w(minionn.VectorInt(overall_w), pkey)

def server_decrypt_u(encU, skey):
    tmp = minionn.VectorInt([])
    minionn.decrypt_w(encU, skey, tmp)
    return tmp

def modulo_pmax(x_in):
    x_in = x_in % config.PMAX

    if abs(x_in) <= config.PMAX_HALF:
        return x_in
    elif x_in > 0:
        return x_in - config.PMAX
    else:
        return x_in + config.PMAX

def client_precomputation(encW, slot_size, w_list):
    """
    Performs the client precomputation.
    This takes the encrypted W from the server and generates
     a v and r for each matrix multiplication.
     r has the shape of x in the W*x multiplication (n x o)
     v has the shape of m x n x o (which gets summed up to n x o later during the client matrix multiplication)
    
    As the r and v values are needed later, they are stored as r0,v0,r1,v1,.. tensors in the tensor dictionary.

    Input:
     - encrypted W
     - slot size
     - w_list: List of tuples:(name of W, dimensions of matrix multiplication [m,n,o])
    Output:
     - encrypted U that can be sent back to the server
    """
    logger.info("Started Client Precomputation.")

    # Use numpy to generate r and v
    client_randoms = []
    for (w,dim) in w_list:
        # Generate v
        v = np.random.randint(config.PMAX, dtype='uint64', size = (dim[0], dim[1], dim[2]))

        if not config.random_v:
            # Allow the construction of a static v in debug mode
            v = np.zeros((dim[0], dim[1], dim[2]), dtype='uint64')

        # Generate r in column major order
        # We will need to transpose r before using it later, but now for precomputation
        #  column major order is required
        r = np.random.randint(config.PMAX, dtype='uint64', size = (dim[2], dim[1]))

        if not config.random_r:
            # Allow the construction of a static r in debug mode
            r = np.multiply(np.ones((dim[2], dim[1]), dtype='uint64'),1)

        client_randoms.append((r,v))

    logger.debug(" - Generated r and v values:")
    for (r,v) in client_randoms:
        logger.debug(" -- r size " + str(r.shape) + " v size " + str(v.shape))

    # Now assemble the big r and v that are used for precomputation
    assembled_R = []
    assembled_V = []
    for i in range(0, len(w_list)): # For every Gemm
        # Assemble R by repeating r_i for every row of W (m times)
        for dm in range(0, w_list[i][1][0]): # For every server row (m) (W row)
            for do in range(0, w_list[i][1][2]): # For every client column o (x col)
                assembled_R.extend(client_randoms[i][0][do].tolist()) # Append a row of r (here, column because it is transposed - Matrix multiplication takes a row times a column)


        # Assemble v by just appending all v's after each other
        assembled_V.extend(client_randoms[i][1].flatten().tolist())

    if config.debug_mode:
        logger.debug(" - Assembled big R: Size " + str(len(assembled_R)) + "; starts with " + str(assembled_R[:config.debug_print_length_long]))
        logger.debug(" - Assembled big V: Size " + str(len(assembled_V)) + "; starts with " + str(assembled_V[:config.debug_print_length_long]))

    # Now we need to transpose the r matrices so that they can be used later (remember, we used r as columns earlier for the matrix multiplication with W)
    logger.debug(" - Transposing r values:")
    for i in range(0,len(client_randoms)):
        # Transpose r
        client_randoms[i] = (client_randoms[i][0].T, client_randoms[i][1])

        # And convert the uint numpy arrays to int cpp arrays for later use
        #  NOTE: We use a modulo with PMAX here to convert from uint to int
        #  This is the same that is done on the cpp side for the homomorphic encryptions.
        #  For the precomputation, Uint64 is needed, and for everything afterwards, int64
        iR = minionn.VectorInt([modulo_pmax(r) for r in client_randoms[i][0].flatten().tolist()])
        _set_tensor("initial_r" + str(i), iR, list(client_randoms[i][0].shape))

        iV = minionn.VectorInt([modulo_pmax(v) for v in client_randoms[i][1].flatten().tolist()])
        _set_tensor("v" + str(i), iV, list(client_randoms[i][1].shape))

        logger.debug(" -- r" + str(i) + " now has size " + str(client_randoms[i][0].shape) + " v" + str(i) + " size " + str(client_randoms[i][1].shape))

    # Generate assembled uint vectors
    uR = minionn.VectorUInt(assembled_R)
    uV = minionn.VectorUInt(assembled_V)

    # Use them for the client precomputation
    encU = minionn.client_precomputation(encW, uR, uV)

    logger.info("Client Precomputation success.")

    # return U
    return encU
    
def extract_sum(inp, dimensions, offset):
    """
    Extracts the sum of the tensor of shape dimension (beginning
    at offset) and returns it.
    dim is assuming a list for [m, n, o] for the matrix calculation mxn * nxo
    This is equal to crow, ccol, srow where server matrix gets multiplied with client matrix
    """
    tmp = minionn.VectorInt([])
    minionn.extract_sum(inp, tmp, 
            dimensions[1], dimensions[2], dimensions[0], 
            offset)

    logger.debug("Extract sum: Extracted with offset " + str(offset)+ " and dimensions " + str(dimensions))
    if config.debug_mode:
        logger.debug("Extracted U starts with " + str(list(tmp)[:config.debug_print_length_long]) + " and ends with " + str(list(tmp)[-config.debug_print_length_long:]))
    return tmp

def vector_add(vec_a, vec_b):
    cpp_a = minionn.VectorInt(vec_a)
    cpp_b = minionn.VectorInt(vec_b)
    return minionn.vector_add(cpp_a, cpp_b)

def vector_sub(vec_a, vec_b):
    cpp_a = minionn.VectorInt(vec_a)
    cpp_b = minionn.VectorInt(vec_b)
    return minionn.vector_sub(cpp_a, cpp_b)

def vector_floor(vector):
    minionn.vector_floor(vector, config.fractional_base)

def matrix_mult(inp, outp, instance_u, order_w_x = True):
    """
    calculates W*x + U + b or
    if order_w_x is False, calculates (W' * X' + U)' + b
    """
    tmp = minionn.VectorInt([])
    
    cpp_w = _get_tensor(inp[0])
    cpp_x = _get_tensor(inp[1])
    cpp_b = _get_tensor(inp[2])

    my_outp = outp

    # Calculate dimensions as a [m,n,o] list
    dims = [
        _get_tensor_dim(inp[0])[0], #first dimension of w
        _get_tensor_dim(inp[0])[1], #second dim of w
        _get_tensor_dim(inp[1])[1] # second dim of x
    ]

            
    if config.debug_mode:
        logger.debug("U is " + str(instance_u))

    b_string = "(b ROW wise)"
    if not order_w_x:
        b_string = "(b COLUMN wise)"

    logger.debug("Performing cpp matrix multiplication " +  b_string + " with " 
        + str(inp) + " to " + my_outp + " with the following dimensions "
        + str(_get_tensor_dim(inp[0])[0]) + "x" + str(_get_tensor_dim(inp[0])[1])
        + " * " + str(_get_tensor_dim(inp[1])[0]) + "x" + str(_get_tensor_dim(inp[1])[1])
    )

    #Compute based on order of W and x
    if order_w_x:   
        # Normal order, calculate W*x + U + b
        minionn.matrixmul(cpp_w,cpp_b,instance_u,cpp_x,dims[1],dims[2],dims[0],tmp)
        # Dimensions are the ones that we were given: 
        #  first dimension of w and second dim of x
    else:
        # Reversed order: (W' * X' + U + b')'
        minionn.matrixmul_b_columns(cpp_w,cpp_b,instance_u,cpp_x,dims[1],dims[2],dims[0],tmp)

        # As we received W' and x', the output now has the dimensions
        #  of the first dimension of x and the second dimension of w (both are transposed)
        # Also, keep in mind that the order is reversed because we store the transposed 
        #  of the final output.
        my_outp = my_outp + "T"

    # Floor the resulting vector to reverse the fractional shifting and store it
    minionn.vector_floor(tmp, pow(config.fractional_base, 1) * config.fractional_downscale)

    _set_tensor(my_outp, tmp, [dims[0], dims[2]])

def matrix_mult_client(inp, outp, v_in, order_w_x = True):
    tmp = minionn.VectorInt([])
    cpp_v = _get_tensor(v_in)
    
    # Calculate dimensions as a [m,n,o] list
    dims = [
        _get_tensor_dim(inp[0])[0], #first dimension of w
        _get_tensor_dim(inp[0])[1], #second dim of w
        _get_tensor_dim(inp[1])[1] # second dim of x
    ]

    logger.debug("Client Gemm with dimensions " + str(dims) + " and actual dims " + str(_get_tensor_dim(inp[0])) + " and " + str(_get_tensor_dim(inp[1])))

    #Compute and store
    minionn.matrixmul_simple(cpp_v,dims[1],dims[2],dims[0],tmp)

    # Floor the resulting vector to reverse the fractional shifting
    minionn.vector_floor(tmp, pow(config.fractional_base, 1) * config.fractional_downscale)

    my_outp = outp
    # If we have a reversed order of operations, we also need to
    # transpose v!!
    if not order_w_x:
        my_outp += "T"

    _set_tensor(my_outp, tmp, [dims[0], dims[2]])

def relu_client(inp, outp, responsible_r):
    # Prepare vectors
    xc = _get_tensor(inp)
    yc = minionn.VectorInt([])
    dims = _get_tensor_dim(inp)
    rc = _get_tensor(responsible_r)

    # Calculate num of elements in vector
    num = reduce(mul, dims, 1)

    # Execute relu
    minionn.relu_client(num, xc, rc, yc)

    # Store ys. Dims did not change
    _set_tensor(outp, yc, dims)

def relu_server(inp, outp):
    # Prepare vectors
    xs = _get_tensor(inp)
    ys = minionn.VectorInt([])
    dims = _get_tensor_dim(inp)

    # Calculate num of elements in vector
    num = reduce(mul, dims, 1)
    
    # Execute relu
    minionn.relu_server(num, xs, ys)

    # Store ys. Dims did not change
    _set_tensor(outp, ys, dims)
