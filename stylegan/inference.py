import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model/checkpoint")
