from subprocess import call
import os
from os.path import isfile, join
from gazelle_utils import *
import tensorflow as tf
import numpy as np

""" Initially 0.00001. Reduce this as we do more epochs. """
LEARNRATE = 0.00001
restarting = False
n_epochs = 10

""" Verification that a file python finds is one we want.
    Return a boolean. """
def verify(filename):
    pass # Put your function here


""" Take in one set of input, hog, labels (e.g. data222.npy, hog222.npy, XYArray222.npy)
    and run the CNN on it. """

def run_one_batch(mode, batch_num_train, batch_num_eval, ep):
    bnt = str(batch_num_train)
    bne = str(batch_num_eval)
    
    print ("Calling the gazecapture_cnn.py command...")
    call("python gazecapture_cnn.py " + mode + ' ' + bnt + ' ' + bne + ' '+ str(LEARNRATE*(0.9**ep))[:10], shell=True)  #[:10] implemented for string format
    
    # We have to figure out some way to log the losses during training, so we can plot it for our report
    
    
    
""" Run one epoch of training (i.e. go through all the training data). """
def run_one_epoch(mode, fileNums, batch_num_eval, ep):
    for f in fileNums:
        print "  ", mode, ": running fileno ", f, "."
        run_one_batch(mode, f, batch_num_eval, ep)

def start_training(instance):
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
    # We train fileNum * epochs (train), 1 * epochs (val), 1 (test)
    payload = np.zeros((epochs*(len(fileNums)+1)+1, 3))
    mod = len(fileNums)+1
    for i in xrange(payload.shape[0]):
        ep = int(i/len(fileNums+1))
        if (i % mod) + 1 == mod:
            payload[i] = [232, 0, valNum, ep]
        if i == payload.shape[0] - 1:
            payload[i] = [233, 0, testNum, ep]
        else:
            payload[i] = [231, (i % mod) + 1, 0, ep]
    np.save("recovering.npy", payload)
    restarting = True
    recover_training("recovering.npy")

def numToMode (num):
    if num is 231:
        return 'train'
    if num is 232:
        return 'val'
    if num is 233:
        return 'test'
    
    
def recover_training(log_file):
    payload = np.load(log_file)
    while (payload.shape[0] > 0):
        args = payload[0,:]
        run_one_epoch(numToMode(args[0]), args[1], args[2], args[3])
        payload = payload[1:,:]
        np.save(log_file, payload)
    print "\n\n    TRAINING HAS FINISHED    \n\n"
    
        
def main(argv):
    if (os.path.isfile("restarting.npy"):
        restarting = True
    
    if restarting is False:
        start_training(argv[1])
    else:
        recover_training("restarting.npy")

if __name__ == "__main__":
    tf.app.run()
