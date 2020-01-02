import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")
sys.path.append('/home/adas/Documents/models/research')
sys.path.append('/home/adas/Documents/models/research/slim')
sys.path.append('/home/adas/Documents/models/research/object_detection')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


PATH_TO_FROZEN_GRAPH ='/home/adas/Documents/output_model/12_16/frozen_inference_graph.pb'
PATH_TO_LABELS = "/home/adas/Documents/output_model/NA_traffic_sign_map_final.pbtxt"
# Path to image
PATH_TO_IMAGE = '/home/adas/Documents/output_model/12_16/test_images'


# Number of classes the object detector can identify
NUM_CLASSES = 100

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # sess = tf.Session(graph=detection_graph)
    config = tf.ConfigProto(log_device_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.6  # don't hog all vRAM
    config.gpu_options.allow_growth = True
    # config.operation_timeout_in_ms = 15000  # terminate on long hangs
    sess = tf.InteractiveSession(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
for root, dir, files in os.walk(PATH_TO_IMAGE):
    for i in files:
        file_path = os.path.join(root, i)
        if ('.png' in file_path or '.PNG' in file_path) and '_labelled' not in file_path:
            print(file_path)
            print('parent folder:',os.path.dirname(file_path))
            image = cv2.imread(file_path)
            original = image
            height, width, _ = original.shape
            # cv2.imwrite(, image)
            print(width, height)
            # We need some resize so the GPU won't crash
            image = cv2.resize(original, (900, 400), interpolation=cv2.INTER_AREA)
            image_expanded = np.expand_dims(image, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.80)

            # All the results have been drawn on image. Now display the image.
            image = cv2.resize(image, (width, height))
            # cv2.imshow('Object detector', image)
            max_score = max(max(scores))
            print('max score', max_score)
            if max_score >= 0.8:
                cv2.imwrite(os.path.dirname(file_path)+'/'+os.path.basename(file_path)[:-3] + "_labelled.png", image)

# Press any key to close the image
# cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
