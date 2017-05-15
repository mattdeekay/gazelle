"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gazelle_utils import *
"""

from face_eye_detector import *

# For now, please run this only in the gazelle/code directory.


"""
Techniques for how to detect faces and eyes drawn from:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
"""

def original_demo():
    # Input and Output file names
    face_in  = "face1_macron.jpg"
    face_out = "face1_macron_out.png"

    face_finder = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
    eye_finder  = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")

    img  = plt.imread(IMG_DIR + face_in, 1) # for display
    mono = plt.imread(IMG_DIR + face_in, 0) # for cascades

    # Detection step.
    faces = face_finder.detectMultiScale(mono, 1.3, 5)
    for (x,y,w,h) in faces:
        # Red bounding box around the face in |img|
        cv2.rectangle(img, (x,y), (x+w, y+h), color=(255,0,0), thickness=2)
        gray_region = mono[y:y+h, x:x+w]
        color_region = img[y:y+h, x:x+w]
        eyes = eye_finder.detectMultiScale(gray_region)
        for (ex,ey,ew,eh) in eyes:
            # Green bounding box around the eyes in |img|
            cv2.rectangle(color_region, (ex,ey), (ex+ew, ey+eh), color=(0,255,0), thickness=2)

    plt.imshow(img)
    plt.savefig(IMG_DIR + face_out)


def detector_class_demo():
    save_name = "face3_feifei_out.png"
    face_in_path  = IMG_DIR + "face3_feifei.jpg"
    face_out_path = IMG_DIR + save_name
    
    detector = Detector(face_in_path, in_color=True, run=True)
    detector.draw_and_save(face_out_path)



if __name__ == "__main__":
    detector_class_demo()


