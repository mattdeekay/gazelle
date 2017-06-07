import numpy as np
#import skimage.io as sio
#from scipy.io import loadmat
#from plotting import *
pic = 16
cib=2
nbins=9

def genhist(im, nbins):
    # The compute gradient function
    dy = im[:-2, 1:-1] - im[2:, 1:-1]
    dx = im[1:-1, :-2] - im[1:-1, 2:]
    mags = np.sqrt(dy * dy + dx * dx)
    angles = 57.2957795 * np.arctan2(dy, dx)
    angles[angles < 0] += 180

    # The original generate_histogram
    hist = np.zeros(nbins)
    binterval = float(180/nbins)
    norm_angle = (angles - 10)/binterval
    bin1 = np.floor(norm_angle).astype(int)
    bin2 = bin1 + 1
    center_angle2 = 10 + 20*bin2
    bin1_amounts = (center_angle2 - angles) / binterval * mags
    bin2_amounts = mags - bin1_amounts
    bin1 = bin1 % nbins
    bin2 = bin2 % nbins
    for m in xrange(angles.shape[0]):
        for n in xrange(angles.shape[1]):
            hist[bin1[m,n]] += bin1_amounts[m,n]
            hist[bin2[m,n]] += bin2_amounts[m,n]
    return hist

def compute_hog_features(im, pic, cib, nbins):

    def bfeat(block):
        return np.array([[genhist(block[y:(y+pic), x:(x+pic)], nbins) for x in xrange(0,cib*pic,pic)] for y in xrange(0,cib*pic,pic)]).flatten()

    # change "xrange" to just "range" f this breaks during tensorflow (because of Python 3)
    window_size = pic * cib
    stride = window_size / 2
    H_blocks, W_blocks = np.array(im.shape) / stride - 1
    hog_feature = []
    for h in xrange(0,H_blocks*stride,stride):
        row = []
        for w in xrange(0,W_blocks*stride,stride):
            beat = bfeat(im[h : h+window_size, w : w+window_size])
            norm = np.linalg.norm(beat)
            beat /= norm if norm != 0 else 1
            row.append(beat)
        hog_feature.append(row)
    return np.array(hog_feature)



if __name__ == '__main__':
    images = ['240.jpg', 'face1_macron.jpg', 'car.jpg']
    """
    for name in images:
        im = sio.imread(name, True) # (height, width) 1-channel image ndarray
        pic = 16
        cib=2
        nbins=9
        poop = compute_hog_features(im, pic, cib, nbins)
        show_hog(im, poop, figsize = (18,6))
    """