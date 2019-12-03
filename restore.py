
import tensorflow as tf
# a) Create the network
saver = tf.train.import_meta_graph('model.ckpt.meta')

# b) Load the parameters

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    