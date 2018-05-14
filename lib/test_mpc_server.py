"""
Author: Fritz Alder
Copyright: 
Secure Systems Group, Aalto University 
https://ssg.aalto.fi/

This code is released under Apache 2.0 license
http://www.apache.org/licenses/LICENSE-2.0
"""
def test_two_vectors(vector, expected_list):
    assert len(list(vector)) == len(expected_list), "Length wrong" + str(len(list(vector))) + " instead of " + str(len(expected_list))
    assert list(vector) == expected_list, "Wrong result: " + str(list(vector)) + " instead of expected " + str(expected_list)

import cppimport
#This will pause for a moment to compile the module
cppimport.set_quiet(False)
m = cppimport.imp("minionn")
print("Successfuly imported c++ code\n")

print("Testing MPC functions, server side...")
m.init_aby("127.0.0.1", 5000, True)
print("Connected to client, testing ReLu.")

num = 5
xs = m.VectorInt([-5,-4,-3,-2,1])

ys = m.VectorInt([])
m.relu_server(num, xs, ys)

print("Relu done, testing correctness.")
#print("Num is " + str(num))
#print("Xs is " + str(xs))
#print("After relu, ys is " + str(ys))

test_two_vectors(ys, [-1, -1, 1, -1, 0])
print("Correct")
print("Testing second run")

num = 5
xs = m.VectorInt([-5,-5,-5,5,5])

ys = m.VectorInt([])
m.relu_server(num, xs, ys)

#print("Num is " + str(num))
#print("Xs is " + str(xs))
#print("After relu, ys is " + str(ys))

test_two_vectors(ys, [-2, -2, -2, 6, 6])
print("Second run correct. Shutting down...")

m.shutdown_aby()

print("All done, test successful.")