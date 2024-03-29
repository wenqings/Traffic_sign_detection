import os
import cv2
import numpy as np
import tensorflow as tf
import sys
sys.path.append("..")
sys.path.append('/home/adas/Documents/models/research')
sys.path.append('/home/adas/Documents/models/research/slim')
sys.path.append('/home/adas/Documents/models/research/object_detection')
# This is needed since the notebook is stored in the object_detection folder.


# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_FROZEN_GRAPH ='/home/adas/Documents/output_model/12_16/frozen_inference_graph.pb'
PATH_TO_LABELS = "/home/adas/Documents/output_model/test_version.pbtxt"
# Number of classes the object detector can identify
NUM_CLASSES = 100

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


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

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    # TODO: ADD Filter here, all of the shape is 1*300
    new_boxes = np.empty((0,4),int)
    new_scores = np.empty((0),float)
    new_classes = np.empty((0),int)
    show_flag = False
    for i in range(300):
        if np.squeeze(scores)[i]>=0.6:
            if np.squeeze(classes).astype(np.int32)[i] in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
                if boxes[0][i][2]-boxes[0][i][0]>=0.01 and boxes[0][i][3]-boxes[0][i][1]>=0.01:
                    new_boxes = np.append(new_boxes,[[boxes[0][i][0],boxes[0][i][1],boxes[0][i][2],boxes[0][i][3]]],axis=0)
                    new_scores = np.append(new_scores,[scores[0][i]],axis=0)
                    new_classes = np.append(new_classes,[classes[0][i]],axis=0)
                    print(classes[0][i])
                    show_flag = True


    print('new_boxes',new_boxes.shape)
    print('new_scores', new_scores.shape)
    print('new_classes', new_classes.shape)

    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.60)
    if show_flag:
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            new_boxes,
            np.array(new_classes).astype(np.int32),
            new_scores,

            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

