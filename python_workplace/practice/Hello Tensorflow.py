import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

