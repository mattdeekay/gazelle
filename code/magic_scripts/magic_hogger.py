from subprocess import call
import os
from os.path import isfile, join
import numpy as np
import hog_module as hog
CURRENT = '.'

def verify(filename):
    return filename[-4:] == ".npy" and filename[:4] == 'data'

def process(filename):
    number = filename[4:-4]
    data = np.load(filename)
    face = data[:,:,:,:,2]
    mono_face = hog.as_monochrome(face)
    hoghist = hog.compute_hog_features(mono_face, pic=12, cib=2, nbins=9)
    print ("hoghist done", filename, hoghist.shape)
    np.save('hog' + number + '.npy', hoghist)

if __name__ == "__main__":
    onlyfiles = [f for f in os.listdir(CURRENT) if isfile(join(CURRENT, f)) and verify(f)]
    
    for f in onlyfiles:
        process(f)

