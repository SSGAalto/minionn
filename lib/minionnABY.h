/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MINIONNABY_H
#define MINIONNABY_H

#include "minionnCommon.h"

//Utility libs
#include "ENCRYPTO_utils/crypto/crypto.h"
#include "ENCRYPTO_utils/parse_options.h"
//ABY Party class
#include "aby/abyparty.h"
#include "circuit/booleancircuits.h"
#include "circuit/arithmeticcircuits.h"
#include "circuit/circuit.h"

extern ABYParty* party;

//for ABY
const uint32_t secparam = 128;
const uint32_t nthreads = 1;
const e_mt_gen_alg mt_alg = MT_OT;
const uint32_t bitlen = 64;


// core Functions for aby
void init_aby(string address, uint16_t port, bool role_is_server);
void shutdown_aby();

// functions for layers
void relu_server(uint32_t num,  vector<Int>* x_s, vector<Int>* y_s);
void relu_client(uint32_t num, vector<Int>* x_c, vector<Int>* r, vector<Int>* y_c);



#endif
