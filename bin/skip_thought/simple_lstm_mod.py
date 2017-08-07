import tensorflow as tf
import numpy as np

class simple_lstm:

    def __init__(self, embeddings, labels):
        self.lr = 0.001
        self.n_inputs = 10
        self.n_steps = 15
        self.n_hidden_units = 20
        self.batch_size = 200
        self.n_classes = 1
        self.training_iters = 5
        self.keep_prob = 0.7

        self.xs_list = embeddings
        self.ys_list = labels

        self.xs = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.ys = tf.placeholder(tf.float32, [None, self.n_classes])

        # define weights
        self.weights = {
            # (10, 128)
            'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units])),
            # (128, 26)
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        self.biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ])),
            # (26, )
            'out': tf.Variable(tf.constant(0.1, shape=[self.n_classes, ]))
        }

    def lstm(self, X, weights, biases):
        # input hidden layer
        X = tf.reshape(X, [-1, self.n_inputs])
        X_in = tf.matmul(X, weights['in'])+biases['in']
        X_in = tf.nn.dropout(X_in, self.keep_prob)
        X_in = tf.reshape(X_in, [self.batch_size, self.n_steps, self.n_hidden_units])

        # cell
        lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

        # output hidden layer
        results_temp = tf.matmul(final_state[1], weights['out']) + biases['out']
        results = tf.nn.dropout(results_temp, self.keep_prob)

        return results

    def train(self):
        pred = tf.nn.sigmoid(self.lstm(self.xs, self.weights, self.biases))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, "answer/saved_net.ckpt")
            step = 0
            result = []
            while step < self.training_iters:
                pointer = step*self.batch_size%len(self.xs_list)
                batch_xs = np.array(self.xs_list[pointer:pointer+self.batch_size])
                batch_xs = batch_xs.reshape([self.batch_size, self.n_steps, self.n_inputs])
                result.append(sess.run(pred, feed_dict={self.xs: batch_xs}))
                step = step+1
        return result
