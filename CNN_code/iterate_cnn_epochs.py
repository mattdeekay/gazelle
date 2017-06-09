from subprocess import call
import os
from os.path import isfile, join
from gazelle_utils import *
import tensorflow as tf

""" Initially 0.00001. Reduce this as we do more epochs. """
LEARNRATE = 0.00001

""" Verification that a file python finds is one we want.
    Return a boolean. """
def verify(filename):
    pass # Put your function here


""" Take in one set of input, hog, labels (e.g. data222.npy, hog222.npy, XYArray222.npy)
    and run the CNN on it. """

def run_one_batch(mode, batch_num_train, batch_num_eval):
    bnt = str(batch_num_train)
    bne = str(batch_num_eval)
    
    call("python gazecapture_cnn.py " + mode + ' ' + bnt + ' ' + bne + ' '+ str(LEARNRATE), shell=True)
    
    # We have to figure out some way to log the losses during training, so we can plot it for our report
    
    
    
""" Run one epoch of training (i.e. go through all the training data). """
def run_one_epoch(mode, fileNums, batch_num_eval):
    for f in fileNums:
        print "  ", mode, ": running fileno ", f, "."
        run_one_batch(mode, f, batch_num_eval)

def main(argv):
    instance = argv[1]
    # Make a list of all the data files inside our folder we need
    from os import listdir
    from os.path import isfile, join
    
    # The folder of data is in Owen's instance
    if (instance == 'o'):
        datapath = "../data_CNN/clean"
    elif (instance == 'm'):
        datapath = "../../../Owen/gazelle-github-Owen/data_CNN/clean"
    onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
    onlyfiles.sort()
    print "There are ", len(onlyfiles), " files: ", onlyfiles
    
    # Grab all the numbers
    import re
    fileNums = []
    for f in onlyfiles:
        fileNums.append(int(re.sub("[^0-9]", "", f)))
    fileNums = list(set(fileNums))
    valNum = fileNums[-2]
    testNum = fileNums[-1]
    fileNums = fileNums[:-2]
    
    n_epochs = 10
    # 'train' 'test' 'validation'
    for i in xrange(n_epochs):
        print "Running epoch ", i, "."
        run_one_epoch('train', fileNums, 0)
        
    run_one_epoch('validation', 0, valNum)
    run_one_epoch('test', 0, testNum)

if __name__ == "__main__":
    tf.app.run()