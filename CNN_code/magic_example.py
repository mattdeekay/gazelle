from subprocess import call
import os
from os.path import isfile, join
import pickle
import numpy as np

CURRENT = '.'
CLEAN = 'clean'
def verify(filename):
    return filename[-7:] == ".npy.gz"

def process(filename):
    rootname = filename[:-7]
    call('gunzip ' + filename, shell=True)


if __name__ == "__main__":
    onlyfiles = [f for f in os.listdir(CURRENT) if isfile(join(CURRENT, f)) and verify(f)]
    
    for f in onlyfiles:
        process(f)

