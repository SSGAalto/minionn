/*
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
*/

#include "minionnMath.h"

/*
  Takes an Int and
   1) calculates x % PMAX
   2) Shifts it with PMAX
*/
Int moduloPMAX(Int x)
{
	x = x % PMAX;
	if(abs(x) <= PMAX_HALF)
	{
		return x;
	}
	else
	{
		if (x > 0)
			return x - PMAX;
		else
			return PMAX + x;
	}
}

vector<Int> vector_add(vector<Int> x, vector<Int> y)
{
	vector<Int> z;
	for(size_t i = 0; i < x.size(); i ++)
		z.push_back(moduloPMAX(x[i]+y[i]));

	return z;
}

vector<Int> vector_sub(vector<Int> x, vector<Int> y)
{
	vector<Int> z;
	for(size_t i = 0; i < x.size(); i ++)
		z.push_back(moduloPMAX(x[i]-y[i]));

	return z;
}

vector<Int> vector_mul(vector<Int> x, Int a)
{
	vector<Int> z;
	for(size_t i = 0; i < x.size(); i ++)
		z.push_back(moduloPMAX(x[i]*a));

	return z;
}

vector<Int> vector_div(vector<Int> x, Int a)
{
	vector<Int> z;
	for(size_t i = 0; i < x.size(); i ++)
	{
		double d = (double)x[i];
		Int t = round(d / a);
		z.push_back(moduloPMAX(t));
	}

	return z;
}

void vector_floor(vector<Int>* x, uInt fractional){
	size_t size = x->size();

  for(size_t i=0; i<size ; i++){
    x->at(i) = Floor(x->at(i), fractional);
  }
}

void vector_raise(vector<Int>* x, uInt fractional){
	size_t size = x->size();

  for(size_t i=0; i<size ; i++){
    x->at(i) = moduloPMAX(x->at(i) * fractional);
  }
}

Int Floor(Int x, uInt f)
{
	if (x >= 0)
		return floor((double) x / f);
	else
		return 0 - floor((double) (0 -x) / f);
}

/*
b is added column wise
u is added element wise

  Wx + b + u
  W has shape m x n
  X has shape n x o
  Result of Wx has shape m x o
  b has shape o
  U has shape m x o
*/
void matrixmul(vector<Int> *W, vector<Int> *b, vector<Int> *U,
vector<Int> *x_s, int nn, int oo, int mm, vector<Int> *y_s)
{
	int c = 0;
	for (int i = 0; i < mm; i ++)
	{
		for (int j = 0; j < oo; j ++)
		{
			Int sum = 0;
			for (int k = 0; k < nn; k ++)
			{
				sum = sum + moduloPMAX((*W)[i*nn + k] * (*x_s)[k*oo + j]);
				sum = moduloPMAX(sum);
			}
			sum = sum + (*b)[j];
			sum = sum + (*U)[c++];
			y_s->push_back(moduloPMAX(sum));
		}
	}
}

/*
Different version of matmul where b is added row wise
u is added element wise

  Wx + b + u
  W has shape m x n
  X has shape n x o
  Result of Wx has shape m x o
  b has shape m
  U has shape m x o
*/
void matrixmul_b_columns(vector<Int> *W, vector<Int> *b, vector<Int> *U,
vector<Int> *x_s, int nn, int oo, int mm, vector<Int> *y_s)
{
	int c = 0;
	for (int i = 0; i < mm; i ++)
	{
		for (int j = 0; j < oo; j ++)
		{
			Int sum = 0;
			for (int k = 0; k < nn; k ++)
			{
				sum = sum + moduloPMAX((*W)[i*nn + k] * (*x_s)[k*oo + j]);
				sum = moduloPMAX(sum);
			}
			sum = sum + (*b)[i];
			sum = sum + (*U)[c++];
			y_s->push_back(moduloPMAX(sum));
		}
	}
}

/*

  just sums v
*/
void matrixmul_simple(vector<Int> *v, int nn, int oo, int mm, vector<Int> *y_c)
{
	int c = 0;
	for (int i = 0; i < mm; i ++)
	{
		for (int j = 0; j < oo; j ++)
		{
			Int sum = 0;
			for (int k = 0; k < nn; k ++)
			{
				sum = moduloPMAX(sum + (*v)[c++]);
			}
			y_c->push_back(sum);
		//	cout << i << " " << y_c[i] << endl;
		}
	}
}

/**
  Fills the out vector with <size> random integers adjusted to PMAX
*/
void generate_random_vector(vector<uInt>* out, int size){
  mt19937_64 rand_gen (std::random_device{}());

  //reserve size for performance
  out->reserve(size);

  for(int i=0; i<size ; i++){
    out -> push_back(rand_gen() % PMAX);
  }
}

/**
* Takes a vector of unsigned Ints (uint64) and returns it as
* Int vector with every number taken as modulo PMAX.
* Note: The documentation of moduloPMAX!
*/
void vector_to_int_PMAX(vector<uInt>* in, vector<Int>* out){
  size_t size = in->size();

  out-> reserve(size);

  for(int i=0; i<size ; i++){
    uInt tmp = in->at(i) % PMAX;
    out -> push_back(moduloPMAX(tmp));
  }
}


/**
  Extracts the sum of a U matrix.
  This is done by reducing the srow x ccol x crow matrix to a srow x ccol matrix
  Only works on a submatrix of the whole array
  for m x n x o matrix this can also be read as
  extract_sum(in,out, o, n, m, offset)
*/
void extract_sum(vector<Int>* in_u, vector<Int>* out_u,
  int crow, int ccol, int srow, int start_pos)
{
  int pos = start_pos;
	for (int i = 0; i < srow; i ++)
	{
		for (int j = 0; j < ccol; j ++)
		{
			Int sum = 0;
			for (int k = 0; k < crow; k ++)
			{
				sum = moduloPMAX(sum + (*in_u)[pos++]);
			}
			out_u->push_back(sum);
		}
	}
}
