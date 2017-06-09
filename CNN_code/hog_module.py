import numpy as np
#import skimage.io as sio
#from plotting import *

def as_monochrome(im):
    # Input |im| = [H, W, 3].
    return im.dot(np.array([0.299, 0.587, 0.114]))


"""
genhist works now with [num, H, W]:
    :return: hist.shape = [nbins, num]
"""
def genhist(im, nbins):
    im = np.lib.pad(im, ((0,0),(1,1),(1,1)), lambda: 1)
    # The compute gradient function
    dy = im[:, :-2, 1:-1] - im[:, 2:, 1:-1]
    dx = im[:, 1:-1, :-2] - im[:, 1:-1, 2:]
    mags = np.sqrt(dy * dy + dx * dx)
    angles = 57.2957795 * np.arctan2(dy, dx)
    angles[angles < 0] += 180

    # The original generate_histogram
    num_im = angles.shape[0]
    hist = np.zeros((nbins, num_im))
    binterval = float(180/nbins)
    norm_angle = (angles - 10)/binterval
    bin1 = np.floor(norm_angle).astype(int)
    bin2 = bin1 + 1
    center_angle2 = 10 + 20*bin2
    bin1_amounts = (center_angle2 - angles) / binterval * mags
    bin2_amounts = mags - bin1_amounts
    bin1 = bin1 % nbins
    bin2 = bin2 % nbins
    for i in xrange(num_im):
        for m in xrange(angles.shape[0]):
            for n in xrange(angles.shape[1]):
                hist[bin1[m,n], i] += bin1_amounts[i,m,n]
                hist[bin2[m,n], i] += bin2_amounts[i,m,n]
    # hist.shape: [nbins, num_im]
    return hist

def compute_hog_features(im, pic, cib, nbins):
    """
    |im| is now [num, H, W].
    Returns ndarray of shape [H_blocks, W_blocks, cib * cib * nbins].
    """
    def bfeat(block):
        beat = np.array([[genhist(block[:, y:(y+pic), x:(x+pic)], nbins) for x in xrange(0,cib*pic,pic)] \
                                                                         for y in xrange(0,cib*pic,pic)])
        # beat.shape = [H=cib, W=cib, nbins, num_im]
        return beat.reshape(-1, beat.shape[3])

    # change "xrange" to just "range" f this breaks during tensorflow (because of Python 3)
    window_size = pic * cib
    stride = window_size / 2
    H_blocks, W_blocks = np.array(im.shape[1:3]) / stride - 1
    
    hog_feature = []
    for h in xrange(0,H_blocks*stride,stride):
        row = []
        for w in xrange(0,W_blocks*stride,stride):

            beat = bfeat(im[:, h:h+window_size, w:w+window_size]) # beat.shape = [cib*cib*nbins, num_im][0]
            norm = np.linalg.norm(beat, axis=0)
            norm[norm == 0] = 1
            beat /= norm
            row.append(beat)
        hog_feature.append(row)
    hog_array = np.array(hog_feature)
    hog_array = np.rollaxis(hog_array, 3) # hog_array.shape = [num_im, H_blocks, W_blocks, cib*cib*nbins]
    return hog_array

if __name__ == '__main__':
    images = ['face1_macron.jpg', 'car.jpg', '240.jpg']
    imarray = []
    for name in images:
        im = sio.imread(name, True) # (height, width) 1-channel image ndarray
        imarray.append(im[:270, 200:470])
    im = np.array(imarray)
    print "starting im", im.shape

    pic = 8
    cib=2
    nbins=9

    poop = compute_hog_features(im, pic, cib, nbins)
    show_hog(im[0], poop[0], figsize = (18,6))
    show_hog(im[1], poop[1], figsize = (18,6))
    show_hog(im[2], poop[2], figsize = (18,6))

