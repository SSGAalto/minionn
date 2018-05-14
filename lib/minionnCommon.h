/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MINIONNCOMMON_H
#define MINIONNCOMMON_H


#include <vector>
#include <sstream>
#include <chrono>
#include "seal/seal.h"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>



#include <math.h>
#include <cassert>
#include <algorithm>
#include <random>
#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/stl_bind.h>
#include <pybind11/include/pybind11/stl.h>
typedef int64_t Int;
typedef uint64_t uInt;

namespace py = pybind11;
using namespace std;
PYBIND11_MAKE_OPAQUE(std::vector<Int>);
PYBIND11_MAKE_OPAQUE(std::vector<uInt>);
//PYBIND11_MAKE_OPAQUE(std::vector<py::bytes>);



const Int PMAX = 101285036033;
const Int PMAX_HALF = PMAX / 2;
extern Int PMAXBITS;
extern int SLOTS;

#endif
