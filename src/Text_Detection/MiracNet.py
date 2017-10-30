import inspect
import os
from OCR.Text_Detection.utils import utils, dataset, pre_process
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import tensorflow.contrib.layers as init

# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--filt", required=True, help="filter")
# ap.add_argument("-lr", "--lr", required=True, help="learning rate")
# args = vars(ap.parse_args())

class MiracNet:
    def __init__(self, name, learning_rate = 0.0064, filter=3, path=None):
        if path is None:
            path = inspect.getfile(MiracNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "text_det.npy")
            print(path)
        self.learning_rate= learning_rate
        self.filter = filter
        self.name = name
        self.ds = dataset.DataSet(source_path=path)

        # Zero-mean features
        [train, val, test ] =utils.pre_process(self.data_dict['train_features'], self.data_dict['val_features'],
                                              self.data_dict['test_features'])
        # Update Data Dictionary
        self.data_dict['train_features'] = train
        self.data_dict['val_features'] = val
        self.data_dict['test_features'] = test
        print("npy file loaded")

    x = tf.placeholder(tf.float32, shape=[None, 6144], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, 2], name='labels')  # One-hot encoded Labels
    y_true_cls = tf.argmax(y_true, dimension=1)
    x_image = tf.reshape(x, [-1, 32, 64, 3])

    sess = tf.Session()
    def ForwardPropagation(self):
        start_time = time.time()
        print("Forward Propagation Started")

        self.conv1_1 = self.new_conv_layer(name='Conv1_1', input=self.x_image, filter_size=self.filter, input_depth=3,
                                           num_of_filters=32, pooling=False)

        self.conv1_2 = self.new_conv_layer(name='Conv1_2', input=self.conv1_1, filter_size=self.filter, input_depth=32,
                                           num_of_filters=32, pooling=True)

        self.conv2_1 = self.new_conv_layer(name='Conv2_1', input=self.conv1_2, filter_size=self.filter, input_depth=32,
                                           num_of_filters=64, pooling=False)

        self.conv2_2 = self.new_conv_layer(name='Conv2_2', input=self.conv2_1, filter_size=self.filter, input_depth=64,
                                           num_of_filters=64,pooling=True)


        layer_flat, num_features = self.flatten_layer(self.conv2_2)
        self.fc1 = self.new_fc_layer(name='Fc1', input=layer_flat, use_relu=True, channels_in=num_features, channels_out=1024)
        self.fc2 = self.new_fc_layer(name='Fc2', input=self.fc1, use_relu=False, channels_in=1024, channels_out=2)

        self.y_pred = tf.nn.softmax(self.fc2)  # Probability Scores of classes for each label
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)  # True Class Labels
        print("right there")
        print(self.y_pred_cls)

        lr = [1e-05, 2e-05, 4e-05, 8e-05, 0.00016, 0.00032, 0.00064, 0.001]

        # Calculate Cost and minimize it with Adam optimizer
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.y_true))

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        with tf.name_scope("accuracy"):
            self.correct_prediction = tf.equal(self.y_pred_cls, self.y_true_cls)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('accuracy', self.accuracy)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('logs/testlogs/' + self.name)
        self.writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver()

    #def BackPropagation(self):


    def optimize(self, num_iterations, train_batch_size):
        # Ensure we update the global variable rather than a local copy.

        # Start-time used for printing time-usage below.
        start_time = time.time()

        startIndex = 0
        finishIndex = train_batch_size
        file = open("testacc_logs.txt", "w")
        total_iterations = 0
        best_validation_accuracy = 0.0
        last_improvement = 0
        require_improvement = 1000
        epoch=0

        for i in range(num_iterations):
            total_iterations+=1
            x_batch, y_true_batch = utils.next_batch(num=train_batch_size, data=self.ds.get(self.ds.train,self.ds.features),
                                                    labels=self.data_dict['train_labels_onehot'],
                                                    start=startIndex, finish=finishIndex)
            feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.sess.run(self.optimizer, feed_dict=feed_dict_train)
            if total_iterations == 600:
                epoch+=1
                startIndex=0
                finishIndex=train_batch_size
                if epoch==10:
                    print("End of 10.th epoch..")
                    break
            else:
                startIndex += train_batch_size
                finishIndex += train_batch_size

            if (i % 100 == 0) or (i == (num_iterations - 1)):
                s = self.sess.run(self.merged_summary, feed_dict=feed_dict_train)
                self.writer.add_summary(s, i)
                test_acc = utils.print_test_accuracy(batch_size=250, x_val=self.data_dict["val_features"],
                                                 y_val=self.data_dict['val_labels'],
                                                 y_val_oh=self.data_dict['val_labels_onehot'],
                                                 x=self.x, y_true=self.y_true, y_pred_cls=self.y_pred_cls,
                                                 sess=self.sess)
                acc_train = self.sess.run(self.accuracy, feed_dict=feed_dict_train)
                acc_validation, msg = test_acc

                # If validation accuracy is an improvement over best-known.
                if acc_validation > best_validation_accuracy:
                    # Update the best-known validation accuracy.
                    best_validation_accuracy = acc_validation

                    # Set the iteration for the last improvement to current.
                    last_improvement = total_iterations

                    # Save all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.sess, save_path='checkpoints/')

                    # A string to be printed below, shows improvement found.
                    improved_str = '*'
                else:
                    # An empty string to be printed below.
                    # Shows that no improvement was found.
                    improved_str = ''

                file.write(msg)
                # Status-message for printing.
                msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

                # Print it.
                print(msg.format(i + 1, acc_train, acc_validation, improved_str))

            if total_iterations - last_improvement > require_improvement:
                print("No improvement found in a while, stopping optimization.")

                # Break out from the for-loop.
                break
        # Update the total number of iterations performed.
        total_iterations += num_iterations

        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        file.close()

    def new_conv_layer(self, input, filter_size, input_depth, num_of_filters, pooling, name):
        with tf.name_scope(name):
            weights = self.new_conv_weights(name='W-{}'.format(name), filterW=filter_size, filterH=filter_size,
                                       input_depth=input_depth, num_of_filter=num_of_filters)
            biases = self.new_conv_biases(name='B-{}'.format(name), num_of_filters=num_of_filters)
            conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
            activated = tf.nn.relu(conv + biases)
            if pooling:
                activated = tf.nn.max_pool(activated, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('biases', biases)
            tf.summary.histogram('activations', activated)
            return activated

    def new_conv_weights(self, name, filterW, filterH, input_depth, num_of_filter):
        return tf.get_variable(name=name, shape=(filterH, filterW, input_depth, num_of_filter),
                               dtype=tf.float32, initializer=init.xavier_initializer())

    def new_conv_biases(self, name, num_of_filters):
        return tf.get_variable(name=name, shape=[num_of_filters],
                               dtype=tf.float32, initializer=init.xavier_initializer())

    def new_fc_weights(self, name, channel_in, channel_out):
        return tf.get_variable(name=name, shape=(channel_in, channel_out),
                               dtype=tf.float32, initializer=init.xavier_initializer())

    def new_fc_biases(self, name, channel_out):
        return tf.get_variable(name=name, shape=[channel_out], dtype=tf.float32,
                               initializer=init.xavier_initializer())

    def flatten_layer(self, last_layer):
        layer_shape = last_layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()

        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(last_layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features

    def new_fc_layer(self, input, use_relu, channels_in, channels_out, name):
        with tf.name_scope(name):
            fc_weights = self.new_fc_weights(name='W-{}'.format(name), channel_in=channels_in, channel_out=channels_out)
            fc_biases = self.new_fc_biases(name='B-{}'.format(name), channel_out=channels_out)  # bug
            layer = tf.matmul(input, fc_weights) + fc_biases
            if use_relu:
                layer = tf.nn.relu(layer)
            return layer

# a = MiracNet()
# a.ForwardPropagation()
#a.optimize(num_iterations=60, train_batch_size=32)

hyperSearch = MiracNet(name="last")
hyperSearch.ForwardPropagation()
# hyperSearch.optimize(num_iterations=1000, train_batch_size=100)

