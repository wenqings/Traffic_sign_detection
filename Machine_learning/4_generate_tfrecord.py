"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import json
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

# flags = tf.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('image_dir', '', 'Path to the image directory')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS
#
flags = tf.app.flags
flags.DEFINE_string('csv_input', 'I:\\RSA\\output\\train_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('image_dir', 'I:\\RSA\\output\\train', 'Path to the image directory')
flags.DEFINE_string('output_path', 'I:\\RSA\\output\\TRAIN_TF\\train.record', 'Path to output TFRecord')
flags.DEFINE_integer('num_shards', 1000, 'Number of TFRecord shards')
FLAGS = flags.FLAGS


# flags = tf.app.flags
# flags.DEFINE_string('csv_input', 'D:\\RSA\\output\\test_labels.csv', 'Path to the CSV input')
# flags.DEFINE_string('image_dir', 'D:\\RSA\\output\\test', 'Path to the image directory')
# flags.DEFINE_string('output_path', 'D:\\RSA\\output\\TEST_TF\\test.record', 'Path to output TFRecord')
# flags.DEFINE_integer('num_shards', 100, 'Number of TFRecord shards')
# FLAGS = flags.FLAGS


with open('D:\\RSA\\output\\NA_traffic_sign_map.json', 'r') as fp:
    json_data = json.load(fp)


# TO-DO replace this with label map
def class_text_to_int(row_label):
   return json_data[row_label]

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    output_path = FLAGS.output_path
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    num_shards = FLAGS.num_shards
    grouped = split(examples, 'filename')
    index = 0
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        # One picture may have multiple traffic sign, so we need to group them
        # One group is one image with multiple(0~x) annotation(s)
        # Each image it will generate a tfRecord format file, we call it tf_example,
        # then we write the tf_example into tfrecord file
        # If we want to save into xxxx(some number) tfrecord file, first image will be save in tfrecorrd_0, second image will be saved into tfrecord_1 ...
        for group in grouped:
            print(index)
            tf_example = create_tf_example(group, path)
            if tf_example:
                shard_idx = index % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            index+=1


if __name__ == '__main__':
    tf.app.run()
