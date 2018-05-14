/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#include "minionnABY.h"
#include "minionnMath.h"

ABYParty* party;

void init_aby(string address, uint16_t port, bool role_is_server){
  e_role role;
  if(role_is_server){
    role = (e_role) 0;
  } else { // client
    role = (e_role) 1;
  }
  seclvl sl = get_sec_lvl(secparam);
  party = new ABYParty(role, (char*) address.c_str(), port, sl, bitlen, nthreads, mt_alg);
}

void shutdown_aby(){
	delete party;
}

void reset_aby(){
	// Simply reset the ABY party object
	party->Reset();
}

void relu_server(uint32_t num,  vector<Int>* x_s, vector<Int>* y_s)
{
	vector<Sharing*>& sharings = party->GetSharings();
	Circuit* y_circ = sharings[S_YAO]->GetCircuitBuildRoutine();
	Circuit* a_circ = sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit* b_circ = sharings[S_BOOL]->GetCircuitBuildRoutine();

	uint64_t* xs = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* xc = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* rc = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* s_out_vec = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint32_t out_bitlen, out_nvals;

	for (int i = 0; i < num; i++) {
		if ((*x_s)[i] >= 0)
			xs[i] = (*x_s)[i];
		else
			xs[i] = PMAX + (*x_s)[i];
	}

	share *xs_share, *xc_share, *rc_share, *x, *sel_share, *N, *halfN, *zero;

	xs_share = y_circ->PutSIMDINGate(num, xs, bitlen, SERVER);
	xc_share = y_circ->PutSIMDINGate(num, xc, bitlen, CLIENT);
	rc_share = y_circ->PutSIMDINGate(num, rc, bitlen, CLIENT);
	N = y_circ->PutSIMDCONSGate(num, PMAX, bitlen);
	halfN = y_circ->PutSIMDCONSGate(num, PMAX/2, bitlen);
	zero = y_circ->PutSUBGate(N, N);

	x = y_circ->PutADDGate(xs_share, xc_share);
	sel_share = y_circ->PutGTGate(N, x);
	x = y_circ->PutMUXGate(x, y_circ->PutSUBGate(x, N), sel_share);
	sel_share = y_circ->PutGTGate(x, halfN);
	x = y_circ->PutMUXGate(zero, x, sel_share);
	x = y_circ->PutADDGate(x, rc_share);
	sel_share = y_circ->PutGTGate(N, x);
	x = y_circ->PutMUXGate(x, y_circ->PutSUBGate(x, N), sel_share);

	x = y_circ->PutOUTGate(x, ALL); //TODO

	party->ExecCircuit();

	x -> get_clear_value_vec(&s_out_vec, &out_bitlen, &out_nvals);

	for(int i = 0; i < num; i ++)
	{
		y_s->push_back(moduloPMAX(s_out_vec[i]));
	//	cout << i << " " << x_s[i] << endl;
	}

	// Reset party
	reset_aby();
}

void relu_client(uint32_t num, vector<Int>* x_c, vector<Int>* r, vector<Int>* y_c)
{
	vector<Sharing*>& sharings = party->GetSharings();
	Circuit* y_circ = sharings[S_YAO]->GetCircuitBuildRoutine();
	Circuit* a_circ = sharings[S_ARITH]->GetCircuitBuildRoutine();
	Circuit* b_circ = sharings[S_BOOL]->GetCircuitBuildRoutine();

	uint64_t* xs = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* xc = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* rc = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint64_t* s_out_vec = (uint64_t*) malloc(num * sizeof(uint64_t));
	uint32_t out_bitlen, out_nvals;

	for (int i = 0; i < num; i++) {
		if ((*x_c)[i] >= 0)
			xc[i] = (*x_c)[i];
		else
			xc[i] = PMAX + (*x_c)[i];

		if ((*r)[i] >= 0) //-r
			rc[i] = PMAX - (*r)[i];
		else
			rc[i] = 0 - (*r)[i];
	}

	share *xs_share, *xc_share, *rc_share, *x, *sel_share, *N, *halfN, *zero;

	xs_share = y_circ->PutSIMDINGate(num, xs, bitlen, SERVER);
	xc_share = y_circ->PutSIMDINGate(num, xc, bitlen, CLIENT);
	rc_share = y_circ->PutSIMDINGate(num, rc, bitlen, CLIENT);
	N = y_circ->PutSIMDCONSGate(num, PMAX, bitlen);
	halfN = y_circ->PutSIMDCONSGate(num, PMAX/2, bitlen);
	zero = y_circ->PutSUBGate(N, N);

	x = y_circ->PutADDGate(xs_share, xc_share);
	sel_share = y_circ->PutGTGate(N, x);
	x = y_circ->PutMUXGate(x, y_circ->PutSUBGate(x, N), sel_share);
	sel_share = y_circ->PutGTGate(x, halfN);
	x = y_circ->PutMUXGate(zero, x, sel_share);
	x = y_circ->PutADDGate(x, rc_share);
	sel_share = y_circ->PutGTGate(N, x);
	x = y_circ->PutMUXGate(x, y_circ->PutSUBGate(x, N), sel_share);

	x = y_circ->PutOUTGate(x, ALL); //TODO

	party->ExecCircuit();

	x -> get_clear_value_vec(&s_out_vec, &out_bitlen, &out_nvals);

	for(int i = 0; i < num; i ++)
	{
		y_c->push_back((*r)[i]);
	//	cout << i << " " << x_s[i] << endl;
	}

	// Reset party
	reset_aby();
}
