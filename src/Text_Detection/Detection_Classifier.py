import tensorflow as tf
import numpy as np


class DetectionClassifier:
    def __init__(self):
        self.session = tf.Session()
        self.graph_path = 'C:/Users/mirac/Desktop/Dev/OCR_PROJECT/src/Text_Detection/checkpoints/trained.meta'
        self.saver = tf.train.import_meta_graph(self.graph_path)
        self.weights = 'C:/Users/mirac/Desktop/Dev/OCR_PROJECT/src/Text_Detection/checkpoints/./'
        self.saver.restore(self.session, tf.train.latest_checkpoint(self.weights))

    def make_predict(self, arr):
        if arr.shape[1] != 6144:
            arr = arr.reshape(1, 6144)
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: arr}
        op_to_restore = graph.get_tensor_by_name("ArgMax_1:0")
        ans = np.asarray(self.session.run(op_to_restore, feed_dict), dtype=np.int)
        return ans[0]
