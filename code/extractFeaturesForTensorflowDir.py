
from Detector import *
import pickle
import json
import scipy.misc

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
    data = None

    for i in range(len(filenames)):
        filename = filenames[i]
        bounds = output[i]
        if bounds is not None:
            image = plt.imread(filename)
            # Getting the face from the bounds
            face = np.array((bounds[0][2], bounds[0][3], 3))
            face = image[bounds[0][1]:(bounds[0][1] + bounds[0][3]), bounds[0][0]:(bounds[0][0] + bounds[0][2])]
            face = scipy.misc.imresize(face, (224, 224, 3))
            face = np.reshape(face, (224, 224, 3, 1))

            # The face grid will be stored in the first layer of the 3D array
            # The face grid represents the location of the face in the image, where the image
            # is squished (symmetry transform?) so it has dimesnions 224x224
            faceGrid = np.zeros((224, 224, 3))
            for i in range(int((bounds[0][0] + 0.0) / image.shape[0] * 224), int((bounds[0][0] + bounds[0][2] + 0.0) / image.shape[0] * 224)):
                for j in range(int((bounds[0][1]  + 0.0)/ image.shape[1] * 224), int((bounds[0][1] + bounds[0][3] + 0.0) / image.shape[1] * 224)):
                    faceGrid[i][j][0] = 1
            faceGrid = np.reshape(faceGrid, (224, 224, 3, 1))

            # Getting the right eye from the bounds
            rightEye = np.array((bounds[1][2], bounds[1][3], 3))
            rightEye = image[bounds[0][1] + bounds[1][1]:(bounds[0][1] + bounds[1][1] + bounds[1][3]), bounds[0][0] + bounds[1][0]:(bounds[0][0] + bounds[1][0] + bounds[1][2])]
            rightEye = scipy.misc.imresize(rightEye, (224, 224, 3))
            leftEye = np.array((bounds[2][2], bounds[2][3], 3))
            leftEye = image[bounds[0][1] + bounds[2][1]:(bounds[0][1] + bounds[2][1] + bounds[2][3]), bounds[0][0] + bounds[2][0]:(bounds[0][0] +bounds[2][0] + bounds[2][2])]
            leftEye = scipy.misc.imresize(leftEye, (224, 224, 3))
            rightEye = np.reshape(rightEye, (224, 224, 3, 1))
            leftEye = np.reshape(leftEye, (224, 224, 3, 1))
            

            # Check using the bounds that the detected right eye is actually to the right
            if (bounds[1][2] < bounds[2][2]):
                entry = np.concatenate((rightEye, leftEye, face, faceGrid), axis=3)
            else:
                entry = np.concatenate((leftEye, rightEye, face, faceGrid), axis=3)
            entry = entry.reshape((224, 224, 3, 4, 1))
    if data is None:
        data = entry
    else:
        data = np.concatenate((data, entry), axis=4)
    output = open(IN_DIR + 'data.pkl', 'wb') 
    pickle.dump(data, output)
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
    output, count = bounds = test_detector_dir_get_bounds()
    generate_arrays_from_output(output, count)
#    generate_XY_from_dir(output)


