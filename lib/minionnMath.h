/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#ifndef MINIONNMATH_H
#define MINIONNMATH_H

#include "minionnCommon.h"

const int fractional = 1000;

Int moduloPMAX(Int x);
Int Floor(Int x, uInt div);
void vector_floor(vector<Int>* x, uInt fractional);
void vector_raise(vector<Int>* x, uInt fractional);

vector<Int> vector_add(vector<Int> x, vector<Int> y);
vector<Int> vector_sub(vector<Int> x, vector<Int> y);
vector<Int> vector_mul(vector<Int> x, Int a);
vector<Int> vector_div(vector<Int> x, Int a);

void matrixmul(vector<Int> *W, vector<Int> *b, vector<Int> *U,
vector<Int> *x_s, int crow, int ccol, int srow, vector<Int> *y_s);
void matrixmul_b_columns(vector<Int> *W, vector<Int> *b, vector<Int> *U,
vector<Int> *x_s, int nn, int oo, int mm, vector<Int> *y_s);

void matrixmul_simple(vector<Int> *v, int nn, int oo, int mm, vector<Int> *y_s);

void generate_random_vector(vector<uInt>* out, int size);
void vector_to_int_PMAX(vector<uInt>* in, vector<Int>* out);

void extract_sum(vector<Int>* in_u, vector<Int>* out_u,
  int crow, int ccol, int srow, int start_pos);

#endif
