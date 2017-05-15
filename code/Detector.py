import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import walk

from gazelle_utils import *
from GazelleError import *


class Detector(object):

    def __init__(self):
        """ self.img_color = image in color, after |load_image|
            self.img_mono  = image in monochrome, after |load_image|
            self.faces
            self.eyes
        """
        self.face_finder = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
        self.eye_finder  = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")


    """
    Use methods below!
    """

    def get_bounds_directory(self, dir_path):
        """ Run self.get_bounds for an entire directory.
            |dir_path| should be relative, and end in '/'.
        """
        # all the files (not dirs) under dir_path that end in .jpg
        filenames = [dir_path + fn for fn in next(walk(dir_path))[2] if fn[-4:] == '.jpg']
        return [self.get_bounds(fn) for fn in filenames]
        """
        output_bounds = []
        counter = 0
        found = 0
        for fn in filenames:
            csp = OUT_TOYFRAMES_DIR + str(counter) + '.png'
            res = self.get_bounds(fn, save_path=csp)
            output_bounds.append(res)
            found += 1 if res is not None else 0
            counter += 1
            # print "done: %s/%s", (found, counter)      # Success rate on toy 01001: 106/629
        return output_bounds
        """



    def get_bounds(self, input_path, save_path=None):
        """ Optional save_path if you want to save the image with
            face and eye bounding boxes.
        """
        display = True if save_path is not None else False

        self._load_image(input_path, get_color=display)
        try:
            bounds = self._get_face_and_eyes(draw=display)
        except GazelleError as gerr:
            # print gerr.message
            return None
        
        if display: self._save_image(save_path)
        return bounds



    #####################################################
    # The Underbelly, don't need to touch this
    #####################################################


    def _load_image(self, input_path, get_color=False):
        """ Get the image from a given file path
        """
        self.input_path = input_path
        if get_color:
            self.img_color = plt.imread(input_path, 1) # for display purposes
        self.img_mono  = plt.imread(input_path, 0) # for cascade algorithm


    def _get_face_and_eyes(self, draw=False):
        """ Returns the face and eye bounding boxes as (x,y,w,h) where
            x,y = coordinates of upper left corner
            w,h = width and height respectively.

            The image is discarded if we detect more than 1 face or more than 2 eyes.
            :return: list of len 3 of (x,y,w,h) tuples for Face, Eye1, Eye2.
        """
        # assert len(self.img_mono.shape) == 3
        # if "img_color" not in self.__dict__:
        #     draw = False

        self.faces = self.face_finder.detectMultiScale(self.img_mono, 1.3, 5)
        if len(self.faces) != 1:
            raise DetectionError("found %s faces in %s" % (len(self.faces), self.input_path))
        
        for x,y,w,h in self.faces:
            face_region = self.img_mono[y:y+h, x:x+w]
            self.eyes = self.eye_finder.detectMultiScale(face_region)
            if len(self.eyes) != 2:
                raise DetectionError("found %s eyes in %s" % (len(self.eyes), self.input_path))

            if draw:
                cv2.rectangle(self.img_color, (x,y), (x+w,y+h), color=(255,0,0), thickness=2)
                face_region_color = self.img_color[y:y+h, x:x+w]
                for ex,ey,ew,eh in self.eyes:
                    cv2.rectangle(face_region_color, (ex,ey),(ex+ew,ey+eh), color=(0,255,0), thickness=2)

        return [tuple(self.faces[0]), tuple(self.eyes[0]), tuple(self.eyes[1])]


    def _save_image(self, save_path):
        """ Saves the image associated with this Detector into directory img/
            with the given name.
        """
        # assert "img_color" in self.__dict__
        plt.imshow(self.img_color)
        plt.savefig(save_path)


