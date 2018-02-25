#!/usr/bin/env python3

import os
import numpy as np
from scipy import misc


############################################################################
# Fuction:  load_mnist                                                     #
# Use:      X, Y = load_mnist(path, kind='train')                          # 
# Inputs:  -path : (string) : path to data directory                       #
#          -kind : (string) : 'test' or 'train'                            #
# Outputs: -X : N x (Mx1) (array) : input data. N instances of Mx1 samples #
#          -Y : N x 1 (array) : N output labels                            #
############################################################################

# Load train and test images as 1D row arrays.
# Load labels as 1D column array
#
def load_mnist(path, kind):
    # Determine path to data and labels
    #
    image_path = os.path.join(path,
                              '%s_data/' % kind)
    label_path = os.path.join(path,
                              'labels/%s_label.txt' % kind)
    
    # Load the output label traget values
    labels = np.loadtxt(label_path,
                        dtype=np.uint8)[:,None]

    # Load the input data values
    #
    images = [misc.imread('%s%s' % (image_path,fn)).flatten() for fn in os.listdir(image_path)]
    images = np.asarray(images)


    return images, labels
