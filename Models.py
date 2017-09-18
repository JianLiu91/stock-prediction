import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn

import tensorflow as tf

def bilinear_attention(source, target):
    # source: [batch_size, n_dim1]
    # target: [batch_size, seq_size, n_dim2]

    # sWt' + b

    dim1 = int(source.get_shape()[1])
    seq_size, dim2 = int(target.get_shape()[1]), int(target.get_shape()[2])

    with tf.variable_scope('attention'):
        W = tf.Variable(tf.truncated_normal([dim1, dim2], 0, 1.0), tf.float32, name="W_att")
        b = tf.Variable(tf.truncated_normal([1], 0, 1.0), tf.float32, name='b_att')

        source = tf.expand_dims(tf.matmul(source, W), 1)
        prod = tf.add(tf.matmul(source, target, adjoint_b=True), b)

        prod = tf.reshape(prod, [-1, seq_size])
        prod = tf.tanh(prod)

        P = tf.nn.softmax(prod)

        probs3dim = tf.reshape(P, [-1, 1, seq_size])
        Bout = tf.matmul(probs3dim, target)
        Bout2dim = tf.reshape(Bout, [-1, dim2])

        return Bout2dim, P


def linear_attention(source, target):
    # source: [batch_size, n_dim1]
    # target: [batch_size, seq_size, n_dim2]

    # W(t:s) + b

    dim1 = int(source.get_shape()[1])
    seq_size, dim2 = int(target.get_shape()[1]), int(target.get_shape()[2])

    with tf.variable_scope('attention'):
        W = tf.Variable(tf.truncated_normal([dim1 + dim2, 1], 0, 1.0), tf.float32, name="W_att")
        b = tf.Variable(tf.truncated_normal([1], 0, 1.0), tf.float32, name='b_att')

        source = tf.ones((1, seq_size, 1)) * tf.expand_dims(source, 1)
        combine = tf.concat([target, source], -1)

        def mul_fn(current_input):
            return tf.matmul(current_input, W) + b

        prod = tf.map_fn(mul_fn, combine)
        prod = tf.tanh(prod)

        P = tf.nn.softmax(tf.reshape(prod, (-1, seq_size)))

        probs3dim = tf.reshape(P, [-1, 1, seq_size])
        out = tf.matmul(probs3dim, target)
        out2dim = tf.reshape(out, [-1, dim2])

        # out2dim =  tf.reduce_sum(tf.expand_dims(P, 2) * target, 1)
        return out2dim



def convolution_text(input_t, filter_sizes, num_filters, name):
    """
        input:   batch, seq_size, embedding_size, channel
        output:  batch, num_filters
    """
    channel = int(input_t.get_shape()[-1])
    seq_size = int(input_t.get_shape()[1])
    embedding_size = int(input_t.get_shape()[2])

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope(name + "conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, channel, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                input_t,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, (seq_size - filter_size + 1), 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(axis=3, values=pooled_outputs)
    return tf.reshape(h_pool, [-1, num_filters_total])


class BasicModel(object):

    def __init__(
        self, news_length, news_count, desc_length, vocab_size,
        embedding_size, hidden, num_classes,
        l2_reg_lambda=0.1, filter_num=50):

        self.input_x = tf.placeholder(tf.int32, [None, news_length * news_count + desc_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        x_news, x_desc = tf.split(self.input_x, [news_length * news_count, desc_length], 1)

        # Embedding layer
        with tf.name_scope("embedding_layer"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W_emb")

            self.embedded_chars_news = tf.nn.embedding_lookup(W, x_news)
            self.embedded_chars_desc = tf.nn.embedding_lookup(W, x_desc)

        self.news_all = tf.reshape(self.embedded_chars_news, [-1, 20, 100])


        def length(sequence):
            used = tf.sign(tf.abs(tf.reduce_sum(sequence, -1)))
            length = tf.reduce_sum(used, axis=1)
            length = tf.cast(length, tf.int32)
            return length

        def _last_relevant(output, length):
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            output_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (np.asarray(length) - 1)
            flat = tf.reshape(output, [-1, output_size])
            relevant = tf.gather(flat, index)
            return relevant


        with tf.variable_scope('Content_LSTM'):
            content_output, state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.BasicLSTMCell(hidden),
                self.news_all,
                dtype=tf.float32
            )

        last_elem = _last_relevant(content_output, length(self.news_all))
        last_elem = tf.reshape(last_elem, [-1, 70, 100])


        self.embedded_chars_desc = tf.expand_dims(self.embedded_chars_desc, -1)
        desc_output = convolution_text(self.embedded_chars_desc, [2, 3], filter_num, 'conv')

        output = linear_attention(desc_output, last_elem)
        output = tf.concat([output, desc_output], axis=1)
        output = tf.tanh(output)

        with tf.name_scope("hidden"):
            W = tf.Variable(tf.random_normal([int(output.get_shape()[1]), hidden]), name='W')
            b = tf.Variable(tf.random_normal([hidden]), name="b")
            output = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            output = tf.nn.dropout(output, self.dropout_keep_prob)


        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal([hidden, num_classes]), name='W')
            b = tf.Variable(tf.random_normal([num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if 'embedding' not in tf_var.name)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.output = self.loss


if __name__ == '__main__':

    print 'Here'
    x = np.ones((5, 1550))
    y = np.ones((5, 2))
    vocab_size = 20

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    session = tf.Session(config=session_conf)

    with session.as_default():

        model = BasicModel(
            20, 70, 150, vocab_size,
            100, 100, 2)

        session.run(tf.global_variables_initializer())
        a = session.run(model.output, feed_dict={model.input_x: x, model.input_y: y, model.dropout_keep_prob: 0.1})
        print a