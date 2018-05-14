/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MINIONNCRYPTO_H
#define MINIONNCRYPTO_H

#include "minionnCommon.h"
using namespace seal;

void init(int slot_count);
void gen_keys(string str_pk, string str_sk);
vector<py::bytes> encrypt_w(std::vector<Int>* in_w, string pkp);
void decrypt_w(vector<string>* w, string skp, vector<Int>* U);
vector<py::bytes> client_precomputation(vector<string>* w_in,
  vector<uInt>* r_in, vector<uInt>* v_in );


#endif
