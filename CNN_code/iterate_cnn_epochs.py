from subprocess import call
import os
from os.path import isfile, join
from gazelle_utils import *

""" Initially 0.00001. Reduce this as we do more epochs. """
LEARNRATE = 0.00001

""" Verification that a file python finds is one we want.
    Return a boolean. """
def verify(filename):
    pass # Put your function here


""" Take in one set of input, hog, labels (e.g. data222.npy, hog222.npy, XYArray222.npy)
    and run the CNN on it. """
def run_one_batch(batch_num_train, batch_num_eval):
    bnt = str(batch_num_train)
    bne = str(batch_num_eval)
    
    call("python gazecapture_cnn.py " + bnt + ' ' + bne + ' ' + LEARNRATE)
    
    # We have to figure out some way to log the losses during training, so we can plot it for our report
    
    
    
""" Run one epoch of training (i.e. go through all the training data). """
def run_one_epoch():
    pass







if __name__ == "__main__":
    
    n_epochs = 2
    for _ in xrange(n_epochs):
        run_one_epoch()
        