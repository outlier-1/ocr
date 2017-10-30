import tensorflow as tf
import numpy as np
import os
sess = tf.Session()
# First let's load meta graph and restore weights

# c = os.getcwd()
# print(c)
x = 'C:/Users/mirac/Desktop/Dev/OCR_PROJECT/src/Text_Detection/checkpoints/trained.meta'
y = 'C:/Users/mirac/Desktop/Dev/OCR_PROJECT/src/Text_Detection/checkpoints/./'
z = 'checkpoints/trained.meta' # FOR LINUX
t = 'checkpoints/./' # FOR LINUX
saver = tf.train.import_meta_graph(x)
saver.restore(sess,tf.train.latest_checkpoint(y))

def makePredict(arr):
    #resize = arr.reshape(1, 6144)
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: arr}
    op_to_restore = graph.get_tensor_by_name("ArgMax_1:0")
    ans = np.asarray(sess.run(op_to_restore, feed_dict),dtype=np.int)
    return ans[0]
