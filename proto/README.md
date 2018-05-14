[Code generation]

General protobuf code generation:
protoc --proto_path=common --python_out=bin/generated common/minionn-onnx.proto  common/onnx-tensor.proto

For grpc generation use:
python3 -m grpc_tools.protoc -I./ --python_out=../common/ --grpc_python_out=../common/ minionn-onnx.proto onnx.proto

BUT:
The problem with onnx is that we want to include the ModelProto and TensorProto objects from
the native onnx python library into our protobuf files

The easiest way to do that was (for me) to make the following changes to the
generated minionn_onnx_pb2 file:
 1) import onnx as onnx__pb2
    instead of importing the actual onnx_pb2 file
 2) In the \_PRECOMPUTATIONRESPONSE.fields and \_COMPUTATIONREQUEST.fields
    change the \_MODELPROTO and \_TENSORPROTO references in onnx to the
    namings in the onnx library: ModelProto and TensorProto
 3) Possibly apply this to all other onnx object references you need.
    this means, for all messages, exchange the generated name for the onnx name
