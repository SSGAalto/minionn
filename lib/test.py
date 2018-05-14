"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""

import cppimport
#This will pause for a moment to compile the module
cppimport.set_quiet(False)
m = cppimport.imp("minionn")
#import minionn as m
print("\nSuccessfuly imported c++ code\n")

SLOTS = 4096
PMAX = 101285036033

import numpy as np
import os
from operator import mul
from functools import reduce

def run_test(shape):
    """
    Here, we just test if the homomorphic encryption works.
    As such, we only test if Dec(Enc(w)*a-c) = w*a-c for every element of w
    """

    # Generate w and encrypt
    w_np = np.random.randint(10000,None,size=shape,dtype='int64')
    w_cpp = m.VectorInt(w_np.flatten().tolist())
    w_cpp = m.VectorInt([i for i in range(0,100)])
    encW = m.encrypt_w(w_cpp,pkey)

    length = reduce(mul, shape, 1)

    r_np = np.random.randint(PMAX, None, size=length, dtype='uint64')
    r = m.VectorUInt(r_np.flatten().tolist())

    v_np = np.random.randint(PMAX,None,size=length, dtype='uint64')
    v = m.VectorUInt(v_np.flatten().tolist())
    
    # Do client precomputation
    encU = m.client_precomputation(encW, r, v)

    # Decrypt w again
    decrypted_u = m.VectorInt([])
    m.decrypt_w(encU, skey, decrypted_u)

    # check if values match with expected value
    ww = list(w_cpp)
    vv = list(v)
    rr = list(r)
    dd = list(decrypted_u)[:length]

    """
    print("W")
    print(ww)
    print("R")
    print(rr[:length])
    print("V")
    print(vv[:length])
    print("D")
    print(dd)
    """
    print("Testing for correctness")
    for i in range(0,length):
        assert dd[i] == m.modulo((ww[i] * rr[i]) - vv[i])
    print("Testing done.")

def test_two_vectors(vector, expected_list):
    assert len(list(vector)) == len(expected_list), "Length wrong" + str(len(list(vector))) + " instead of " + str(len(expected_list))
    assert list(vector) == expected_list, "Wrong result: " + str(list(vector)) + " instead of expected " + str(expected_list)


## Maths tests
print("### Basic maths tests")
a = m.VectorInt([1,2])
b = m.VectorInt([3,4])
c = m.VectorInt([4,6])
d = m.VectorInt([10000000000,20000000000,30000000000,35000000000,-21000000000])
e = m.VectorInt([1,2,-2])
null_matrix = m.VectorInt([0,0,0,0])
null_vector = m.VectorInt([0,0])
print("Testing vector operations")
test_two_vectors(m.vector_add(a,b), [4,6])
test_two_vectors(m.vector_sub(a,b), [-2,-2])
test_two_vectors(m.vector_mul(b,3), [9,12])
test_two_vectors(m.vector_div(c,2), [2,3])
m.vector_floor(d,10000000000)
test_two_vectors(d,[1,2,3,3,-2])
m.vector_raise(e,10000000000)
test_two_vectors(e,[10000000000,20000000000,-20000000000])

w = m.VectorInt([1,2,3,4])
x = m.VectorInt([4,3,2,1])
u = m.VectorInt([2,5,0,7])
b = m.VectorInt([20,10])
y = m.VectorInt([])
print("Testing matrix multiplication")
print("Normal matmul (b broadcasted)")
m.matrixmul(w,b,u,x,2,2,2,y)
test_two_vectors(y, [30,20,40,30])
print("Row wise matmul (b.T broadcasted)")
y = m.VectorInt([])
m.matrixmul_b_columns(w,b,u,x,2,2,2,y)
test_two_vectors(y, [30,30,30,30])

print("Testing extract sum")
dim_m = 10
dim_n = 5
dim_o = 6
a = [i%(dim_m*dim_n) for i in range(0,dim_m*dim_n*dim_o)]
a = sorted(a)
a_vec = m.VectorInt(a)
b_vec = m.VectorInt([])

#Test all
m.extract_sum(a_vec, b_vec, dim_o, dim_n, dim_m, 0)
b_baseline = [dim_o * i for i in range(0,dim_m*dim_n)]
test_two_vectors(b_vec, b_baseline)

#Create subset behind a and test it
new_m = 2
new_n = 2
new_o = 3
a.extend(sorted([i%(new_m*new_n) for i in range(0,new_m*new_n*new_o)]))
b_baseline = [new_o * i for i in range(0,new_m*new_n)]
a_vec = m.VectorInt(a)
b_vec = m.VectorInt([])
m.extract_sum(a_vec, b_vec, new_o, new_n, new_m, dim_m*dim_n*dim_o)
test_two_vectors(b_vec, b_baseline)

## Crypto tests
#crypto operations return a list of bytes
print("### Homomorphic + precomputation tests")
asset_folder = "assets/"

if not os.path.exists(asset_folder):
    os.makedirs(asset_folder)
    print("Created directory " + asset_folder)

pkey = asset_folder + "s.pkey"
skey = asset_folder + "s.skey"

shape = (10,10)

# Init library and generate keys
m.init(SLOTS)
m.gen_keys(pkey, skey)

print("Running simple encrypt/decrypt example")
sample = m.VectorInt([1,2,3,4,5,6,7,8,7,6,5,4,-12,-14])
encW = m.encrypt_w(sample,pkey)
decrypted = m.VectorInt([])
m.decrypt_w(encW, skey, decrypted)
test_two_vectors(sample, list(decrypted)[:len(list(sample))])

print("Running homomorphic test with random r and v")
run_test(shape)

print("Cleanup")
os.remove(pkey)
os.remove(skey)
try:
    os.rmdir(asset_folder)
except os.OSError as identifier:
    print("Not removing non-empty directory " + asset_folder)

print("### All tests passed")
