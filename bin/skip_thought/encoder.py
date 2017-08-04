import tensorflow as tf
import numpy as np

class lstm_encoder(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, learning_rate):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name = 'xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name = 'ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('lstm_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def add_input_layer(self):
        
