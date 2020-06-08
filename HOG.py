import dlib

window = dlib.image_window()

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 4
options.be_verbose = True

detector1 = dlib.fhog_object_detector("output/detector.svm")
detector2 = dlib.fhog_object_detector("output/detector.svm")
detectors = [detector1, detector2]

images = dlib.load_rgb_image('test_images/test_7.jpg')
[boxes, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple(detectors, images, upsample_num_times=1,
                                                                             adjust_threshold=0.0)
for i in range(len(boxes)):
    print("detector {} found box {} with confidence {}.".format(detector_idxs[i], boxes[i], confidences[i]))
images = [dlib.load_rgb_image("test_images/test_7.jpg")]
boxes_img1 = ([dlib.rectangle(left=142, top=246, right=605, bottom=708)])
boxes = [boxes_img1]

detector2 = dlib.train_simple_object_detector(images, boxes, options)
window.set_image(detector2)
dlib.hit_enter_to_continue()