# Commonly used things for Gazelle

# Directory paths:

# Root directory of the repo.
# G_ROOT = "/afs/.ir/users/o/j/ojwang/cs231a/Gazelle/" # Only for running on corn
G_ROOT = "../" # Only if code is run from the /code directory
G_ARCHIVE = "../archive/"

# Location of Haar cascade xml files.
HAAR_DIR = G_ARCHIVE + "opencv/haarcascades/"

# Outputted data from our own code.
OUT_DATA_DIR = G_ARCHIVE + "out_faces_01001/"

# Images to test and run on.
TEST_IMG_DIR = G_ARCHIVE + "test_faces/"
OUT_IMG_DIR = G_ARCHIVE + "out_faces/"

TOYFRAMES_DIR = G_ROOT + "toy_dataset/01001/frames/"
OUT_TOYFRAMES_DIR = OUT_DATA_DIR + "toy_frames_01001/"

CNN_DATA_ROOT = G_ROOT + "toy_CNN_data/"
