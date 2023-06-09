import tensorflow as tf


class CharCNN(object):
    def __init__(self, alphabet_size, document_max_len, num_class):
        self.learning_rate = 1e-4
        self.filter_sizes = [7, 7, 3, 3, 3, 3]
        self.num_filters = 256
        self.kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.05)
        

        self.x = tf.compat.v1.placeholder(tf.float32, [None, document_max_len, 6, 1], name="x") #changed this
        # print(type(self.x))
        # quit()
        self.y = tf.compat.v1.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.compat.v1.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)
        # self.x_tensor = tf.Tensor(value = self.x, dtype=tf.float32, shape=[None, 1014, 70])
        # print(self.x.shape)
        # quit()
        # self.x_one_hot = tf.one_hot(self.x, alphabet_size)
        
        # self.x_expanded = tf.expand_dims(self.x, -1)
        
        # print(type(self.x_one_hot))
        # print(document_max_len)
        # print(type(self.x_one_hot))
        # print('shape')
        # print(self.x_expanded.shape)
        # print(type(self.x_expanded))
        # quit()
        
        
        
        # tf.print(self.x_one_hot)
        # ============= Convolutional Layers =============
        with tf.name_scope("conv-maxpool-1"):
            conv1 = tf.compat.v1.layers.conv2d(
                self.x,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[0], 6],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool1 = tf.compat.v1.layers.max_pooling2d(
                conv1,
                pool_size=(3, 1),
                strides=(3, 1))
            pool1 = tf.transpose(pool1, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-2"):
            conv2 = tf.compat.v1.layers.conv2d(
                pool1,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[1], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool2 = tf.compat.v1.layers.max_pooling2d(
                conv2,
                pool_size=(3, 1),
                strides=(3, 1))
            pool2 = tf.transpose(pool2, [0, 1, 3, 2])

        with tf.name_scope("conv-3"):
            conv3 = tf.compat.v1.layers.conv2d(
                pool2,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[2], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv3 = tf.transpose(conv3, [0, 1, 3, 2])

        with tf.name_scope("conv-4"):
            conv4 = tf.compat.v1.layers.conv2d(
                conv3,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[3], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv4 = tf.transpose(conv4, [0, 1, 3, 2])

        with tf.name_scope("conv-5"):
            conv5 = tf.compat.v1.layers.conv2d(
                conv4,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[4], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            conv5 = tf.transpose(conv5, [0, 1, 3, 2])

        with tf.name_scope("conv-maxpool-6"):
            conv6 = tf.compat.v1.layers.conv2d(
                conv5,
                filters=self.num_filters,
                kernel_size=[self.filter_sizes[5], self.num_filters],
                kernel_initializer=self.kernel_initializer,
                activation=tf.nn.relu)
            pool6 = tf.compat.v1.layers.max_pooling2d(
                conv6,
                pool_size=(3, 1),
                strides=(3, 1))
            pool6 = tf.transpose(pool6, [0, 2, 1, 3])
            h_pool = tf.reshape(pool6, [-1, 34 * self.num_filters])

        # ============= Fully Connected Layers =============
        with tf.name_scope("fc-1"):
            fc1_out = tf.compat.v1.layers.dense(h_pool, 1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)

        with tf.name_scope("fc-2"):
            fc2_out = tf.compat.v1.layers.dense(fc1_out, 1024, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)

        with tf.name_scope("fc-3"):
            self.logits = tf.compat.v1.layers.dense(fc2_out, num_class, activation=None, kernel_initializer=self.kernel_initializer)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        # ============= Loss and Accuracy =============
        with tf.name_scope("loss"):
            self.y_one_hot = tf.one_hot(self.y, num_class)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_one_hot))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
            
        
#         print('Layer 1')
#         print(conv1)
#         print(pool1)
#         print('\nLayer 2')
#         print(conv2)
#         print(pool2)
#         print('\nLayer 3')
#         print(conv3)
#         # print(pool3)
#         print('\nLayer 4')
#         print(conv4)
#         # print(pool4)
#         print('\nLayer 5')
#         print(conv5)
#         print('\nLayer 6')
#         print(conv6)
#         print(pool6)
#         print('\nFully Connected 1')
#         print(fc1_out)
#         print('\nFully Connected 2')
#         print(fc2_out)
#         print('\nOutput')
#         print(self.predictions.shape)
# #         print(pool1)
# #         print('in char_cnn')
# #         quit()
#         # b = sess.run(model.conv1)
#         # print(b)
#         quit()
