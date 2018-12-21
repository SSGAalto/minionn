"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
import sys, time, os

# ONNX/Numpy
import onnx
import onnx.numpy_helper
import numpy as np

#gRPC for client-server communication
import grpc

# cpp
import cppimport
import cppimport.import_hook
cppimport.set_quiet(True)
import lib.minionn as minionn

#project imports
from common import minionn_onnx_pb2
from common import minionn_onnx_pb2_grpc
from common import onnx_helper, operation_handler, minionn_helper, config

# Logging
import logging
import logging.config
logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger('minionn')

def main():
    parser = argparse.ArgumentParser(description="MiniONN - ONNX compatible version")
    parser.add_argument(
        "-i", "--input",
        type=str, required=True,
        help="The input file for the client's input. Should contain the X vector as a single TensorProto object.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str, required=False,
        help="If given, prints the output matrix into this file (csv comma style).",
    )
    parser.add_argument(
        "-s", "--server",
        type=str, required=False, default=config.ip,
        help="IP address of the server.",
    )
    parser.add_argument(
        "-p","--rpc_port",
        type=int, required=False, default=config.port_rpc,
        help="Server port for MPC.",
    )
    parser.add_argument(
        "-m","--mpc_port",
        type=int, required=False, default=config.port_aby,
        help="Server port for MPC.",
    )
    parser.add_argument(
        "-v", "--verbose",
        required=False, default=False, action='store_true',
        help="Log verbosely.",
    )
    args = parser.parse_args()

    """
    Create and set up Logger
    """
    loglevel = (logging.DEBUG if args.verbose else logging.INFO)
    logger.setLevel(loglevel)
    logger.info("MiniONN CLIENT")

    """
    First, read the x vector from input
    """
    x = onnx.TensorProto()
    with open(args.input, 'rb') as fid:
        content = fid.read()
        x.ParseFromString(content)

    if len(x.dims) == 0:
        logger.error("Error reading the ONNX tensor. Aborting.")
        sys.exit()

    x_list = onnx_helper.onnx_tensor_to_list(x)
    # multiply by fractional
    x_list = [int(config.fractional_base*v) for v in x_list]   

    logger.info("Successfuly read X from input.")
    if config.debug_mode:
        logger.debug("Input starts with " + str(x_list[:config.debug_print_length]) + " and has size " + str(len(x_list)))

    """
    With x ready, we can connect to the server to receive the model and w
    """
    channel = grpc.insecure_channel(args.server + ":" + str(args.rpc_port), options=config.grpc_options)
    stub = minionn_onnx_pb2_grpc.MinioNNStub(channel)
    response = stub.Precomputation(minionn_onnx_pb2.PrecomputationRequest(request_model=True, request_w=True))
    server_w = response.w
    server_model = response.model

    logger.info("Server sent privatized model.")

    # Parse model and fill in dimensions
    tensors_dims = onnx_helper.retrieveTensorDimensionsFromModel(server_model)
    nodes = onnx_helper.retrieveNodesFromModel(server_model)

    for name,dim in tensors_dims.items():
        minionn_helper.put_cpp_tensor(name, None, dim)

    """
    Create handler for the model.
    The handler will already calculate the dimensions of all tensors
    """
    handler = operation_handler.OperationHandler(nodes, server_model.graph.input[0].name)

    """
    Init and generate keys
    """
    if not os.path.exists(config.asset_folder):
        os.makedirs(config.asset_folder)
        logger.info("Created directory " + config.asset_folder)

    minionn_helper.init(config.SLOTS)
    minionn_helper.generate_keys(config.client_pkey,config.client_skey)

    """
    Use w to generate u and xs
    """
    start_time = time.time()
    w_list = handler.get_w_list()
    encU = minionn_helper.client_precomputation(server_w, config.SLOTS, w_list)

    """
    Initialize the handler as client
    This generates the input r that we can use to calculate xs
    """
    handler.init_client()

    # Calculate xs = x - r. Use the input r 
    #   (Activation functions later obliviously put in the next Rs)
    # The input r might be different from the first r due to transposes/reshapes done on x
    input_r = "r0" # First r calculated by init_client
    xs = minionn_helper.vector_sub(x_list, minionn_helper.get_cpp_tensor(input_r))
    # Client input is just the first v
    xc = minionn_helper.get_cpp_tensor(input_r)
    
    """
    Request a computation result from server for u and xs
    and start execution locally.
    """
    result_future = stub.Computation.future(minionn_onnx_pb2.ComputationRequest(u=encU, xs=xs))
    logger.info("Sent Computation request to server.")
    if config.debug_mode:
        logger.debug("x is:" + str(x_list[:config.debug_print_length_long]))
        logger.debug("xs is:" + str(xs[:config.debug_print_length_long]))
        logger.debug("xc is:" + str(xc[:config.debug_print_length_long]))
        logger.debug("input r is:" + str(minionn_helper.get_cpp_tensor("r0")[:config.debug_print_length_long]))

    # Connect to MPC port
    logger.info("Establishing MPC connection")
    minionn_helper.init_mpc(args.server, args.mpc_port, False)

    # Now run the model with xc
    result_client = handler.run_network(x_in = xc,
        in_name = server_model.graph.input[0].name,
        out_name = server_model.graph.output[0].name) 

    logger.info("Shutting down MPC connection")
    minionn_helper.shutdown_mpc()
    
    # Get server result and calculate final result
    result_server = result_future.result().ys
    logger.info("Server result is:" + str(result_server))
    logger.info("Client result is:" + str(result_client))
    

    result = minionn_helper.vector_add(result_client, result_server)
    logger.info("Overall result is: " + str(list(result)))
    finish_time = time.time()
    logger.info("Processing took " + str(finish_time - start_time) + " seconds.")

    # Output to file if requested
    if args.output:
        # reshape to numpy array first
        shape = minionn_helper.get_cpp_tensor_dim(server_model.graph.output[0].name)
        reshaped = np.array(result).reshape(shape)
        # use numpy to store file
        np.savetxt(args.output,reshaped, delimiter=",")

if __name__ == '__main__':
    main()
