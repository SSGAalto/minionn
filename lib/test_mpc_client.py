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

print("Testing MPC functions, client side...")
m.init_aby("127.0.0.1", 5000, False)
print("Connected to server, testing ReLu.")

num = 5
xc = m.VectorInt([1,2,5,0,0])
rc = m.VectorInt([1,1,1,1,1])

yc = m.VectorInt([])
m.relu_client(num, xc, rc, yc)

print("Relu done, testing correctness.")
#print("Num is " + str(num))
#print("Xc is " + str(xc))
#print("Rc is " + str(rc))
#print("After relu, yc is " + str(yc))
test_two_vectors(yc, [1, 1, 1, 1, 1])
print("Correct")
print("Testing second run")

num = 5
xc = m.VectorInt([2,2,3,3,3])
rc = m.VectorInt([2,2,2,2,2])

yc = m.VectorInt([])
m.relu_client(num, xc, rc, yc)

#print("Num is " + str(num))
#print("Xc is " + str(xc))
#print("Rc is " + str(rc))
#print("After relu, yc is " + str(yc))

test_two_vectors(yc, [2, 2, 2, 2, 2])
print("Second run correct. Shutting down...")

m.shutdown_aby()

print("All done, test successful.")