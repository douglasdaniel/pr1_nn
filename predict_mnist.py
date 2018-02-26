#!/usr/bin/env python3

import pdb
import aux
import mxnet as mx
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mxnet import gluon, nd, autograd


# Set context
cntx = mx.cpu()

# Define data path
data_path = 'data/'

# Define path to weights file 'multiclass_parameters.txt'
output_path = 'out/multiclass_parameters.txt'

# import test data
X_test, Y_test = aux.load_mnist(data_path,'test')

# instantiate data iterator. We will test 10 random samples
test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_test,Y_test),
                                  10, shuffle=True)

# Load the trained weights and biases
prams = pickle.load( open(output_path, "rb") )
for i, W in enumerate(prams):
    prams[i] = mx.nd.array(W, ctx=cntx)

# Sample 10 random points from the test set
#
for i, (data, label) in enumreate(test_data):
    data = data.as_in_context(cntx)
    print("Shape of data is <%s>" % data.shape)
    im = nd.transpose(data,(1,0,2,3))
    im = nd.reshape(im,(28,10*28,1))
    tiles = nd.tile(im, (1,1,3))
    plt.imshow(tiles.asnumpy())
    plt.show()
    y_out = aux.nnet(data.reshape(-1,784),prams[0],prams[1])
    y_labels = nd.argmax(y_out, axis=1)
    print("Model classifications asre: ", y_labels)
    break


