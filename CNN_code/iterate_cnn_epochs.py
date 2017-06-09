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

    
    
    
    
""" Run one job (row). Can be training, validation, or testing. """
def execute_one_row(mode, batch_num_train, batch_num_eval, ep):
    print ("  ", mode, ": epoch", ep, "running fileno ", batch_num_train, ".")
    
    bnt = str(batch_num_train)
    bne = str(batch_num_eval)
    
    print ("Calling the gazecapture_cnn.py command...")
    returncode = call("python gazecapture_cnn.py " + mode + ' ' + bnt + ' ' + bne + ' '+ str(LEARNRATE*(0.9**ep)), shell=True)
    return returncode
    


def start_training(instance):
    """
    Either create the .npy configuration, or read it in.
    """
    global n_epochs
    global restarting
    
    # The folder of data is in Owen's instance
    if (instance == 'o'):
        datapath = "../data_CNN/clean"
    elif (instance == 'm'):
        datapath = "../../../Owen/gazelle-github-Owen/data_CNN/clean"
    onlyfiles = [f for f in os.listdir(datapath) if isfile(join(datapath, f))]
    onlyfiles.sort()
    print ("There are ", len(onlyfiles), " files: ", onlyfiles)
    
    
    ###############################
    # Grab all the numbers.
    import re
    fileNums = []
    for f in onlyfiles:
        fileNums.append(int(re.sub("[^0-9]", "", f)))
    fileNums = list(set(fileNums))
    print (fileNums)
    
    valNum = 2
    print ("valNum set aside:", valNum)
    testNum = 4
    print ("testNum set aside:", testNum)
    fileNums.remove(2)
    fileNums.remove(4)
    
    print ("file numbers, after two set aside:", fileNums)
    # We train fileNum * epochs (train), 1 * epochs (val), 1 (test)
    
    epoch_size = len(fileNums)+1
    print ("epoch_size", epoch_size)
    all_with_test = n_epochs*(epoch_size)+1 # testing job += 1
    payload = np.zeros((all_with_test, 4)).astype(int)

    for i in range(payload.shape[0]):
        ep = int(i/epoch_size)
        if (i % epoch_size) + 1 == epoch_size:
            payload[i] = np.array([232, 0, valNum, ep]) # validation
        # The last "job"
        elif i == payload.shape[0] - 1:
            payload[i] = np.array([233, 0, testNum, ep]) # test
        else:
            payload[i] = np.array([231, fileNums[i % epoch_size], 0, ep]) # train
            
    np.save("restarting_iter.npy", payload)
    restarting = True
    recover_training("restarting_iter.npy")

def numToMode (num):
    if num == 231:
        return 'train'
    elif num == 232:
        return 'validation'
    elif num == 233:
        return 'test'
    # return None
    
    
def recover_training(log_file):
    payload = np.load(log_file)
    while (payload.shape[0] > 0):
        args = payload[0,:] # first row
        print ("args", args)
        
        retcode = execute_one_row(numToMode(args[0]), args[1], args[2], args[3])
        print ("We got a return code of", retcode)
        if retcode != 0:
            print ("Error: return code from most recent gazecapture call not 0, quitting. Please debug.")
            quit()

        payload = payload[1:,:] # remove first row
        np.save(log_file, payload)
    print ("\n\n    TRAINING HAS FINISHED    \n\n")
    
        
def main(argv):
    global restarting
    
    if (os.path.isfile("restarting_iter.npy")):
        restarting = True
    
    if not restarting:
        start_training(argv[1])
    else:
        recover_training("restarting_iter.npy")

if __name__ == "__main__":
    tf.app.run()
