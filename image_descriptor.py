from aligner import face_aligner
import dlib
import cv2


def image_description(image):

    # models to landmarks detection and descriptor computing
    predictor_model = "neuro/shape_predictor_68_face_landmarks.dat"
    face_recognition = "neuro/dlib_face_recognition_resnet_model_v1.dat"

    # file_name = image

    # Create a HOG face detector using the built-in dlib class
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_rec = dlib.face_recognition_model_v1(face_recognition)

    face_descriptors = []

    # return list of list with aligned faces witch was in image and their shapes
    aligned_images_data = face_aligner(image)

    # if no faces found function return 0
    if aligned_images_data == 0:
        return 0

    # Get manually bounding box to face to improve kod performance
    for i, aligned_image in enumerate(aligned_images_data["images"]):
        cv2.imwrite('output/temp_image.jpg', aligned_image)
        image = cv2.imread('output/temp_image.jpg')
        height, width, channels = image.shape
        rectangle = dlib.rectangle(left=0, top=0, right=width, bottom=height)

        # There we get face descriptor
        shape = face_pose_predictor(aligned_image, rectangle)
        face_descriptors.append(face_rec.compute_face_descriptor(aligned_image, shape))

    # data with descriptors and bounding boxes
    image_data = {"shapes": aligned_images_data["shapes"], "descriptors": face_descriptors}
    return image_data
