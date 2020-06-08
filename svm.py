from sklearn.preprocessing import LabelEncoder
from image_descriptor import image_description
from sklearn.svm import SVC
import pickle
import glob
import cv2
import os


def get_image_descriptors_from_path(folderImagePaths):

    # initialize our lists of extracted facial embeddings and corresponding people names
    knownDescriptors = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    for folderImagePath in folderImagePaths:
        imagePaths = glob.glob(folderImagePath + "/*.jpg")
        print imagePaths

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):

            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            print imagePath
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            imageData = image_description(image)

            # add the name of the person + corresponding face embedding to lists
            knownNames.append(name)
            knownDescriptors.append(imageData["descriptors"][0])
            total += 1

    # return data with descriptors
    data = {"descriptors": knownDescriptors, "names": knownNames}
    return data


def train_svm(folderImagePaths):
    data = get_image_descriptors_from_path(folderImagePaths)

    # encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["descriptors"], labels)

    # write the actual face recognition model to disk
    f = open("output/recognizer", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open("output/label_encoder", "wb")
    f.write(pickle.dumps(le))
    f.close()
