
from Detector import *
import pickle
import json

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
    faceArray = []
    leftEyeArray = []  
    rightEyeArray = []
    faceBoundsArray = []

    for i in range(len(filenames)):
        filename = filenames[i]
        bounds = output[i]
        if bounds is not None:
            image = plt.imread(filename)
            face = np.array((bounds[0][2], bounds[0][3], 3))
            face = image[bounds[0][1]:(bounds[0][1] + bounds[0][3]), bounds[0][0]:(bounds[0][0] + bounds[0][2])]
            faceBounds = np.zeros(4)
            faceBounds[0] = bounds[0][0] - image.shape[0]/2
            faceBounds[1] = bounds[0][0] + bounds[0][2] - image.shape[0]/2
            faceBounds[2] = bounds[0][1] - image.shape[1]/2
            faceBounds[3] = bounds[0][1] + bounds[0][3] - image.shape[1]/2
            faceBoundsArray.append(faceBounds)
            faceArray.append(face) 
            rightEye = np.array((bounds[1][2], bounds[1][3], 3))
            rightEye = image[bounds[0][1] + bounds[1][1]:(bounds[0][1] + bounds[1][1] + bounds[1][3]), bounds[0][0] + bounds[1][0]:(bounds[0][0] + bounds[1][0] + bounds[1][2])]
            leftEye = np.array((bounds[2][2], bounds[2][3], 3))
            leftEye = image[bounds[0][1] + bounds[2][1]:(bounds[0][1] + bounds[2][1] + bounds[2][3]), bounds[0][0] + bounds[2][0]:(bounds[0][0] +bounds[2][0] + bounds[2][2])]
            if (bounds[1][2] < bounds[2][2]):
                rightEyeArray.append(rightEye)
                leftEyeArray.append(leftEye)
            else:
                leftEyeArray.append(rightEye)
                rightEyeArray.append(leftEye)
    output = open(IN_DIR + 'faceArray.pkl', 'wb') 
    pickle.dump(faceArray, output)
    output.close()
    output = open(IN_DIR + 'leftEyeArray.pkl', 'wb') 
    pickle.dump(leftEyeArray, output)
    output.close()
    output = open(IN_DIR + 'rightEyeArray.pkl', 'wb') 
    pickle.dump(rightEyeArray, output)
    output.close()
    output = open(IN_DIR + 'faceBoundsArray.pkl', 'wb') 
    pickle.dump(faceBoundsArray, output)
    output.close()
            
def generate_XY_from_dir(output):        
    indices = [fn for fn in next(walk(IN_DIR))[2] if fn[-4:] == '.jpg']
    indices = [int(i[0:i.find('.')]) for i in indices]
    XYArray = []
    with open(IN_DIR + "../dotInfo.json") as data_file:    
        data = json.load(data_file)
    
   
    for i in range(len(indices)):
        index = indices[i]
        if output[i] is not None:
            XYArray.append((int(data["XPts"][index]), int(data["YPts"][index])))
    output = open(IN_DIR + 'XYArray.pkl', 'wb') 
    pickle.dump(XYArray, output)
    output.close()
            
        

if __name__ == "__main__":
#    original_demo()
    output, count = bounds = test_detector_dir_get_bounds()
    generate_arrays_from_output(output, count)
    generate_XY_from_dir(output)


