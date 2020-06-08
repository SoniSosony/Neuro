import glob

from image_descriptor import image_description
from svm import train_svm
import numpy as np
import imutils
import pickle
import cv2

# folderImagePaths = ["images\People\Barack Obama", "images\People\Chad Smith", "images\People\Chan Kong-sang", 'images\People\Chester Bennington', 'images\People\Eden Hazard', "images\People\Elizabeth Woolridge Grant", 'images\People\Elon Musk', 'images\People\Gianluigi Buffon', "images\People\Gordon Ramsay", 'images\People\Hideo Kojima', 'images\People\Ivan Urgant', "images\People\Jenifer Aniston", "images\People\Jimmy Fallon", 'images\People\Linus Torvalds', 'images\People\Margaret Qualley', 'images\People\Monica Bellucci', 'images\People\Robyn Rihanna Fenty', 'images\People\Till Lindemann', 'images\People\Will Ferrell', 'images\People\Will Smith']
folderImagePaths = ['images\People\Will Ferrell', 'images\People\Jenifer Aniston']
# train_svm(folderImagePaths)

folderImagePaths = 'images\People\Barack Obama\Additional'

# path to test image
imagePaths = 'test_images/test_7.jpg'

# load the test image
image = cv2.imread(imagePaths, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=600)

# load the actual face recognition model along with the label encoder
loadedRecognizer = pickle.loads(open("output/recognizer", "rb").read())
le = pickle.loads(open("output/label_encoder", "rb").read())

# Recognize the persons on the image using SVM
inputImageDescriptors = image_description(image)

# For each face in image detect the person
for i, test in enumerate(inputImageDescriptors['descriptors']):
    inputImageDescriptor = np.array(inputImageDescriptors["descriptors"][i])
    inputImageDescriptor = inputImageDescriptor.reshape(1, -1)
    preds = loadedRecognizer.predict_proba(inputImageDescriptor)
    j = np.argmax(preds)
    proba = preds[0][j]
    name = le.classes_[j]

    shape = inputImageDescriptors["shapes"][i]
    (startX, startY, endX, endY) = (shape.left(), shape.top(), shape.right(), shape.bottom())

    # draw the bounding box of the face along with the associated probability
    text = "{}".format(name)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    print proba
    if proba > 0.1:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

image_folder = 'video/frames_Monica/'

img_array = []

frame_path = glob.glob(image_folder + '/*.jpg')
frame_count = len(frame_path) - 1

for i in range(0, frame_count):
    frame_path = image_folder + str(i) + '.jpg'
    img = cv2.imread(frame_path)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('Monica_det.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

cv2.imshow("Image", image)
cv2.waitKey(0)
