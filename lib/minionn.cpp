/*cppimport
<%
setup_pybind11(cfg)
cfg['dependencies'] = ['minionnCommon.h','minionnMath.h', 'minionnCrypto.h', 'minionnABY.h']
cfg['libraries'] = [
    #SEAL library
    'seal',
    #Aby library
    'bin/aby',
    #Utilities required for ABY and Miracl
    'ssl', 'crypto', 'gmp', 'gmpxx', 'pthread'
]
cfg['sources'] = [
    # Minionn sources
    'minionnMath.cpp',
    'minionnCrypto.cpp',
    'minionnABY.cpp',
]
cfg['include_dirs'] = ['SEAL', 'ABY/src/abycore', '/usr/lib']
cfg['parallel'] = True
cfg['compiler_args'] = ['-std=c++17']
%>
*/

/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#include "minionnCommon.h"
#include "minionnMath.h"
#include "minionnCrypto.h"
#include "minionnABY.h"

PYBIND11_MODULE(minionn, m) {
    py::bind_vector<std::vector<Int>>(m, "VectorInt", py::buffer_protocol());
    py::bind_vector<std::vector<uInt>>(m, "VectorUInt", py::buffer_protocol());
  //  py::bind_vector<std::vector<py::bytes>>(m, "VectorBytes", py::buffer_protocol());
//    py::bind_vector<std::vector<std::string>>(m, "VectorStr", py::buffer_protocol());
    m.doc() ="C++ module for MiniONN";
    /*
    First, the math module
    */
    m.def("modulo", &moduloPMAX, "Modulo the given number to PMAX");
    m.def("floor", &Floor, "Floors a number");
    m.def("vector_floor", &vector_floor,  "Floors a whole vector (divided by fractional) in place");
    m.def("vector_raise", &vector_raise,  "Raises a whole vector by the given fractional in place");
    m.def("vector_div", &vector_div, "Vector scalar division");
    m.def("vector_add", &vector_add, "Vector to vector addition");
    m.def("vector_sub", &vector_sub, "Vector to vector subtraction");
    m.def("vector_mul", &vector_mul, "Vector scalar mult");
    m.def("matrixmul", &matrixmul, "Multiplies the Inputs according to y = Wx + b + U");
    m.def("matrixmul_b_columns", &matrixmul_b_columns, "Multiplies the Inputs according to y = Wx + b + U. b is added column wise");
    m.def("matrixmul_simple", &matrixmul_simple, "Multiplies the Inputs according to y = Wx");
    m.def("generate_random_vector", &generate_random_vector, "Generates a random UNSIGNED int vector of given size");
    m.def("vector_to_int_PMAX", &vector_to_int_PMAX, "Takes a vector of unsigned int and returns it as a vector of signed int, adjusted to PMAX");
    m.def("extract_sum", &extract_sum, "Reduces a part of a 3-dim matrix to 2-dims");

    /*
    Next, the crypto functions
    */
    m.def("init", &init, "Initializes important parameters for the crypto lib");
    m.def("gen_keys", &gen_keys, "Generates the keys required for homomorphic encryption");
    m.def("encrypt_w",&encrypt_w, "Takes W and encrypts it with the server pkey");
    m.def("decrypt_w", &decrypt_w, "Takes an encrypted w as string vector and outputs its decryption to given vector");
    m.def("client_precomputation", &client_precomputation, "Client precomputation according to MiniONN. Takes encrypted w, r, and v and returns U");

    /*
    Lastly, the ABYcore dependent functions.
    These are mainly functions for all layers of the conv nn
    */
    m.def("init_aby", &init_aby, "Initializes important parameters for the ABY MPC");
    m.def("shutdown_aby", &shutdown_aby, "Shuts down the ABY MPC");

    m.def("relu_server", &relu_server, "Server version of ReLU");
    m.def("relu_client", &relu_client, "Client version of ReLU");
}
