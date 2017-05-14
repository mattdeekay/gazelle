import numpy as np
import cv2
from gazelle_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Detector(object):

    def __init__(self):
        self.face_finder = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
        self.eye_finder  = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")

    def _load_image(self, input_path, color=False):
        if color:
            self.img_color = plt.imread(input_path, 1) # for display purposes
        self.img_mono  = plt.imread(input_path, 0) # for cascade algorithm

    def _get_face(self):
        """
        Returns the face and eye bounding boxes as (x,y,w,h) where
        x,y = coordinates of upper left corner
        w,h = width and height respectively.

        The image is discarded if we detect more than 1 face or more than 2 eyes.
        :return: list of len 3 of (x,y,w,h) tuples,
                 or None if data is noisy.
        """
        faces = self.face_finder.detectMultiScale(self.mono, 1.3, 5)
        
        if len(faces) != 1: return "[Face Error: len %s]" % len(faces)

        fx, fy, fw, fh = faces[0]
        region = self.mono[fy : fy+fh, fx : fx+fw]
        eyes = self.eye_finder.detectMultiScale(region)
        
        if len(eyes) != 2: return "[Eye Error: len %s]" % len(eyes)

        boxes = [(fx, fy, fw, fh)] + eyes
        assert len(boxes) == 3
        return boxes

# +========================

    def get_face_and_eyes(self, input_path):
        """
        User facing function.
        """
        _load_image(input_path)
        return _get_face()
