import dlib

predictor_model = "neuro/shape_predictor_68_face_landmarks.dat"


def face_aligner(img):

    # Load face detector model
    face_detector = dlib.get_frontal_face_detector()

    # Load shape predictor model to find face landmarks
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = face_detector(img, 1)

    print dets

    num_faces = len(dets)
    if num_faces == 0:
        print("[ERROR] Sorry, there were no faces found ")
        return 0

    print("[INFO]Face or faces were found")

    #window = dlib.image_window()

    landmarks = dlib.full_object_detections()
    shapes = []
    total = 0

    # Find the face landmarks
    for detection in dets:
        # Function sp return face landmarks found in bounding boxes
        shape = face_pose_predictor(img, detection)
        shapes.append(detection)
        landmarks.append(shape)

        total += 1

        #window.set_image(img)
        #window.clear_overlay()
        #window.add_overlay(shape)
        #dlib.hit_enter_to_continue()

    if total == 1:
        # It is also possible to get a single chip
        images = [dlib.get_face_chip(img, landmarks[0], size=320)]
        #window.clear_overlay()
        #window.set_image(images[0])
        #raw_input("Press Enter to continue...")
    else:
        # Get the aligned face images
        images = dlib.get_face_chips(img, landmarks, size=320)
        #for image in images:
            #window.clear_overlay()
            #window.set_image(image)
            #raw_input("Press Enter to continue...")

    data = {"shapes": shapes, "images": images}
    return data
