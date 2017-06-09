from subprocess import call
import os
from os.path import isfile, join
import numpy as np
import pickle
CURRENT = '.'

def verify(filename):
    return filename[-4:] == ".pkl"

def process(filename):
    rootname = filename[:-4]
    po = pickle.load(open(filename, 'rb'))
    

    np.save(rootname + '.npy', po)

if __name__ == "__main__":
    onlyfiles = [f for f in os.listdir(CURRENT) if isfile(join(CURRENT, f)) and verify(f)]
    
    for f in onlyfiles:
        process(f)

