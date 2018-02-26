#!/usr/bin/env python3
from mxnet import nd



############################################################################
# Fuction:  softmax                                                        #
# Use:      sig_z = softmax(z)                                             # 
# Inputs:  -z : (array) : linearly valued data array                       #
# Outputs: -sig_z : (array) : normalized probabalistic valued data         #
############################################################################

# Multiclass logistic regression function
#
def softmax(z):
    exp = nd.exp(z - nd.max(z, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))

    return exp / norms


def nnet(X,W,W0):
    y = nd.dot(X,W) + W0
    y_hat = softmax(y)

    return y_hat


def cross_ent(Y_hat, Y):
    return -nd.sum(Y * nd.log(Y_hat + 1e-6))


def SGD(prams, learn_rate):
    for parameter in prams:
        parameter[:] = parameter - learn_rate*parameter.grad
