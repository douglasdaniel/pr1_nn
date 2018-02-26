#!/usr/bin/env python3

# Author : Daniel Douglas
# Date   : 2018/02/10

from mxnet import nd
import pdb

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


############################################################################
# Fuction:  nnet                                                           #
# Use:      y_hat = nnet(X,W,W0)                                           # 
# Inputs:  -X : (-1xD) (array) : input values                              #
#          -W : (DxK) (array) : parameter weights                          #
#          -W0 : (1xK) (array) : parameter biases                          #
# Outputs: -y_hat : (1xK) (array) : normalized probability for each class  #
############################################################################
# Linear neural net with single non-input layer
#
def nnet(X,W,W0):
    y = nd.dot(X,W) + W0
    y_hat = softmax(y)

    return y_hat


############################################################################
# Fuction:  cross_ent                                                      #
# Use:      loss = cross_ent(Y_hat, Y)                                     # 
# Inputs:  -Y_hat : (1xK) (array) : normailized probability for each class #
#          -Y : (1xK) (array) : 'one-hot' encoded true output label        #
# Outputs: -loss : float32 (scalar) : loss function output                 #
############################################################################
# Cross-entropy loss function
#
def cross_ent(Y_hat, Y):
    return - nd.sum(Y * nd.log(Y_hat+1e-6))


############################################################################
# Fuction:  SGD                                                            #
# Use:      prams = SGD(prams, learn_rate)                                 # 
# Inputs:  -prams : (D+1 x K )(array) : weights and biases                 #
#          -learn_rate : (scalar) : learning rate                          #
#                                                                          #
# Outputs: -prams : () (array) : weights updated using SGC                 #
############################################################################
# Stochastic gradient descent updating of parameters
#
def SGD(prams, learn_rate):
    for parameter in prams:
        parameter[:] = parameter - learn_rate*parameter.grad

        return prams

############################################################################
# Fuction:  compute_accuracy                                               #
# Use:      accuracy = compute_accuracy(data_iter, model, weights, cntx)   #
# Inputs:  -data_iter : (data iterator) : input data iterator              #
#          -model : (function) : network impletation function, e.g. nnet() #
#          -weights : (D+1 x K)(array) : weights and biases                #
#          -cntx : (computation context) : GPU or CPU                      #
# Outputs: -accuracy : float32 (scalar) : decimal accuracy of mode         #
############################################################################
# Evaluate model performance over an entire dataset
#
def compute_accuracy(data_iter, model, weights, cntx):
    num = 0
    den = 0
    for i, (X, Y) in enumerate(data_iter):
        data = X.as_in_context(cntx).reshape((-1,X.shape[1]))
        label = Y.as_in_context(cntx)
        Y_hat = model(data,weights[0],weights[1])
        classified = nd.argmax(Y_hat, axis=1).reshape((-1,1))
        num += nd.sum(classified == label)
        den += data.shape[0]
    return (num / den).asscalar()
