import numpy as np
import cv2
from gazelle_utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Detector(object):

    def __init__(self, input_path, in_color=False, run=False, save=False):
        """ self.img_color = image in color, after |load_image|
            self.img_mono  = image in monochrome, after |load_image|
            self.boxes = 3-list of (x,y,w,h) tuples for face, eye1, eye2 after |get_face| is run
        """
        self.img_mono = None
        self.img_color = None
        self.boxes = None
        self.face_finder = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
        self.eye_finder  = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")
        load_image(input_path, in_color)


    def load_image(self, input_path, in_color=False):
        if in_color:
            self.img_color = plt.imread(input_path, 1) # for display purposes
        self.img_mono  = plt.imread(input_path, 0) # for cascade algorithm


    def get_face_and_eyes(self):
        """ Returns the face and eye bounding boxes as (x,y,w,h) where
            x,y = coordinates of upper left corner
            w,h = width and height respectively.

            The image is discarded if we detect more than 1 face or more than 2 eyes.
            :return: list of len 3 of (x,y,w,h) tuples,
                     or None if data is noisy.
        """
        assert len(self.img_mono.shape) == 3

        faces = self.face_finder.detectMultiScale(self.img_mono, 1.3, 5)
        if len(faces) != 1: return "[Face Error: len %s]" % len(faces)

        fx, fy, fw, fh = faces[0]
        face_region = self.img_mono[fy : fy+fh, fx : fx+fw]
        eyes = self.eye_finder.detectMultiScale(face_region)
        if len(eyes) != 2: return "[Eye Error: len %s]" % len(eyes)

        self.boxes = [(fx, fy, fw, fh)] + eyes # check this - actually 3-list of 4-tuples?
        assert len(self.boxes) == 3
        return self.boxes


    def draw_and_save(self, save_path):
        """ Saves the image associated with this Detector into directory img/
            with the given name.
        """
        assert len(self.img_color.shape) == 3
        assert len(self.boxes) == 3
        
        fx, fy, fw, fh = self.boxes[0]
        # red box around the face
        cv2.rectangle(self.img_color, (fx,fy), (fx+fw, fy+fh), color=(255,0,0), thickness=2)
        # green boxes around the eyes
        for x,y,w,h in self.boxes[1:]:
            cv2.rectangle(self.img_color, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)

        plt.imshow(self.img_color)
        plt.savefig(save_path)


