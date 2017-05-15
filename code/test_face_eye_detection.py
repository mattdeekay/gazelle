
from Detector import *

# For now, please run this only in the gazelle/code directory.


"""
Techniques for how to detect faces and eyes drawn from:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
"""

in_names = ["face1_macron.jpg", "face2_legend.jpg", "face3_feifei.jpg"]

def original_demo():
    # Input and Output file names
    face_in  = "face1_macron.jpg"
    face_out = "face1_macron_out.png"

    face_finder = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_frontalface_default.xml")
    eye_finder  = cv2.CascadeClassifier(HAAR_DIR + "haarcascade_eye.xml")

    img  = plt.imread(TEST_IMG_DIR + face_in, 1) # for display
    mono = plt.imread(TEST_IMG_DIR + face_in, 0) # for cascades

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
    plt.savefig(OUT_IMG_DIR + face_out)


def detector_class_demo():
    """
    for fname in in_names:

        face_in_path  = TEST_IMG_DIR + fname
        face_out_path = OUT_IMG_DIR + fname + "_out.png"

        detector = Detector(face_in_path, draw=True, run=True)
        detector.save_image(face_out_path)
    """
    pass

def detector_filter_num_face_eyes_demo():
    """ Discard images with != 1 face and != 2 eyes. Can't be sure those were identified correctly.
        feifei.jpg should be the only one that doesn't satisfy this.
    """
    """
    for fname in in_names:

        face_in_path  = TEST_IMG_DIR + fname
        face_out_path = OUT_IMG_DIR + fname + "_out.png"

        detector = Detector(face_in_path, draw=True, run=True)
        detector.save_image(face_out_path)
        # console output: "found 6 faces in ../test_img/face3_feifei.jpg"
        # As expected: feifei output in OUT_IMG_DIR with no boxes
    """
    pass

def detector_new_pipeline():
    """ decomposed the pipeline. All tests before this now unusable.
    """
    detector = Detector()
    in_names = ['240.jpg']

    for fname in in_names:
        in_path  = TEST_IMG_DIR + fname
        out_path = OUT_IMG_DIR + fname + "_out.png"
        try:
            # Only the first one outputs images to the (out_img/) dir.
            found = detector.get_bounds(in_path)
            # found = detector.get_bounds(in_path, save_path=out_path)
            
            print "found:", found #3-list of 4-tuples
        except GazelleError as gerr:
            print gerr.message
            continue

def test_detector_dir_get_bounds():
    """ test the dir get bounds method """
    detector = Detector()
    output = detector.get_bounds_directory(TOYFRAMES_DIR)
    actual = sum([1 for a in output if a is not None])
    
    gold = 629
    print "result %s/%s." % (actual, gold)
    print output[:10]



if __name__ == "__main__":
    test_detector_dir_get_bounds()


