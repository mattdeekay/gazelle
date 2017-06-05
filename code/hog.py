import numpy as np
import skimage.io as sio
from scipy.io import loadmat
import math
import matplotlib.pyplot as plt
from scipy.misc import imrotate

# Displays the HoG features next to the original image
def show_hog(orig, w, figsize = (8,6)):
    w = np.tile(w, [1, 1, 3])
    w = np.pad(w, ((0,0), (0,0), (0,5)), 'constant', constant_values=0.0)

    #  # Make pictures of positive and negative weights
    pos = hog_picture(w)
    neg = hog_picture(-w)

    # Put pictures together and draw
    buff = 10
    if w.min() < 0.0:
        pos = np.pad(pos, (buff, buff), 'constant', constant_values=0.5)
        neg = np.pad(neg, (buff, buff), 'constant', constant_values=0.5)
        im = np.hstack([pos, neg])
    else:
        im = pos

    im[im < 0] = 0.0
    im[im > 1] = 1.0
    plt.figure(figsize = figsize)
    plt.subplot(121)
    plt.imshow(orig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(im, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.save()


# Make picture of positive HOG weights.
def hog_picture(w, bs = 20):
    # construct a "glyph" for each orientaion
    bim1 = np.zeros((bs, bs))
    bim1[:,int(round(bs/2.0))-1:int(round(bs/2.0))+1] = 1.0
    bim = np.zeros((bs, bs, 9))
    bim[:,:,0] = bim1
    for i in xrange(1,9):
        bim[:,:,i] = imrotate(bim1, -i * float(bs), 'nearest') / 255.0

    # make pictures of positive weights bs adding up weighted glyphs
    w[w < 0] = 0.0
    im = np.zeros((bs * w.shape[0], bs * w.shape[1]))
    for i in xrange(w.shape[0]):
        for j in xrange(w.shape[1]):
            for k in xrange(9):
                im[i * bs : (i+1) * bs, j * bs : (j+1) * bs] += bim[:,:,k] * w[i,j,k+18]

    scale = max(w.max(), -w.max()) + 1e-8
    im /= scale
    return im


def compute_gradient(im):
    H, W = im.shape
    angles = np.zeros((H-2, W-2))
    magnitudes = np.zeros((H-2, W-2))

    for x in range(H-2):
        for y in range(W-2):
            angle=np.arctan2((im[x,y+1]-im[x+2,y+1]),(im[x+1,y]-im[x+1,y+2]))*(180/math.pi)
            if angle < 0:
                angle += 180
            angles[x,y] = angle
            magnitudes[x,y]=np.sqrt((im[x,y+1]-im[x+2,y+1])**2+(im[x+1,y]-im[x+1,y+2])**2)

    return angles, magnitudes

def generate_histogram(angles, magnitudes, nbins = 9):
    histogram = np.zeros(nbins)
    center_angles = np.zeros(nbins)
    for i in xrange(nbins):
        center_angles[i] = (0.5+i) * 180 / nbins

    M, N = angles.shape
    for m in xrange(M):
        for n in xrange(N):
            angle = angles[m, n]
            magnitude = magnitudes[m, n]
            bin_diff = np.abs(center_angles - angle)

            # angle near 0 degrees
            if (180 - center_angles[-1] + angle) < bin_diff[1]:
                bin_diff[-1] = 180 - center_angles[-1] + angle
            # angle near 180 degrees
            if (180 - angle + center_angles[0]) < bin_diff[-2]:
                bin_diff[0] = 180 - angle + center_angles[0]

            ca1, ca2 = np.argsort(bin_diff)[:2]
            histogram[ca1] += magnitude * bin_diff[ca2] / (180.0/nbins)
            histogram[ca2] += magnitude * bin_diff[ca1] / (180.0/nbins)

    return histogram

def compute_hog_features(im, pixels_in_cell, cells_in_block, nbins):
    angles, magnitudes = compute_gradient(im)
    cell = np.zeros((pixels_in_cell, pixels_in_cell))
    block = np.zeros((cells_in_block, cells_in_block))
    block_size = pixels_in_cell * cells_in_block
    stride = block_size / 2

    H, W = im.shape
    H_blocks = (H - block_size) / stride + 1
    W_blocks = (W - block_size) / stride + 1

    features = np.zeros((H_blocks, W_blocks, cells_in_block * cells_in_block * nbins))

    for h in xrange(H_blocks):
        for w in xrange(W_blocks):
            block_angles = angles[h*stride : h*stride+block_size, w*stride : w*stride+block_size]
            block_magnitudes = magnitudes[h*stride : h*stride+block_size, w*stride : w*stride+block_size]
            block_histogram = np.zeros((cells_in_block, cells_in_block, nbins))

            for x in xrange(cells_in_block):
                for y in xrange(cells_in_block):
                    cell_angles = block_angles[x*pixels_in_cell : (x+1)*pixels_in_cell, \
                    y*pixels_in_cell : (y+1)*pixels_in_cell]
                    cell_magnitudes = block_magnitudes[x*pixels_in_cell : (x+1)*pixels_in_cell, \
                    y*pixels_in_cell : (y+1)*pixels_in_cell]
                    block_histogram[x, y] = generate_histogram(cell_angles, cell_magnitudes, nbins)

            block_histogram = block_histogram.flatten() 
            features[h, w] = block_histogram / np.linalg.norm(block_histogram)
    return features


if __name__ == '__hog__':
	#####				TO DO					#####
	# These are hyperparameters that we need to decide on
	pixels_in_cell = 8
    cells_in_block = 2
    nbins = 9
    #####				TO DO					#####
    # We will need to import all images into an array of images here
    features =  []
    for im in images:
    	feat = compute_hog_features(im, pixels_in_cell, cells_in_block, nbins)
    	features.append(feat)
    	# The below might be buggy or slow.
    	save_hog(im, feat, figsize = (18,6)) # We'll need to change figsize



