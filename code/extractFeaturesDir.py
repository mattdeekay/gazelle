
from Detector import *

# For now, please run this only in the gazelle/code directory.


"""
Techniques for how to detect faces and eyes drawn from:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
"""

in_names = ["face1_macron.jpg", "face2_legend.jpg", "face3_feifei.jpg"]
IN_DIR = G_ROOT + "toy_dataset/01000/frames/small_test/"

def test_detector_dir_get_bounds():
    """ test the dir get bounds method """
    detector = Detector()
    output = detector.get_bounds_directory(IN_DIR)
    actual = sum([1 for a in output if a is not None])
    
    return output, actual

def generate_arrays_from_output(output, count):
    filenames = [IN_DIR + fn for fn in next(walk(IN_DIR))[2] if fn[-4:] == '.jpg']
    face = []
    lefteye = []  

    for i in range(len(filenames)):
        filename = filenames[i]
        bounds = output[i]
        if bounds is not None:
            image = plt.imread(filename)
            face = np.array((bounds[0][2], bounds[0][3], 3))
            face = image[bounds[0][0]:bounds[0][2], bounds[0][1]: bounds[0][3]]
            plt.imshow(face)
            plt.show()
            
        
        
        

if __name__ == "__main__":
#    original_demo()
    output, count = bounds = test_detector_dir_get_bounds()
    generate_arrays_from_output(output, count)


