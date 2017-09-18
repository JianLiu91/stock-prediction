# -*- coding: utf8
import os
import sys
import datetime
import tensorflow as tf
from data_helper import *

from Models import BasicModel as MyModel

print 'pid', os.getpid()
print MyModel.__name__

# Parameters
# ==================================================
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Model Hyperparameters

tf.flags.DEFINE_float("dev_sample_percentage", 0.005, "Dev split")

tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 99999, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 30, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("random_seed", 42, "")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


tf.flags.DEFINE_boolean("continue_flag", False, "")
tf.flags.DEFINE_string("dir", "runs", "")
tf.flags.DEFINE_boolean("test_mode", False, "")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
# Load data
print("Loading data...")
x, y, vocab_size = read_data()

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
train_x, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
train_y, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

test_x = x_dev
test_y = y_dev

print("Train/Dev/Test : {:d}/{:d}/{:d}".format(len(train_y), len(test_y), len(test_y)))

out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.dir))
print("\n\nWriting to {}\n".format(out_dir))

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        tf.set_random_seed(FLAGS.random_seed)

        # Load user sepicific model
        with tf.variable_scope('model', initializer=tf.contrib.layers.xavier_initializer()):
            model = MyModel(20, 70, 150, vocab_size, 100, 100, 2)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-6)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")

        # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            # _, step, summaries, loss, accuracy, p, r, f1, count = sess.run(
            #     [train_op, global_step, train_summary_op, model.loss, model.accuracy, model.p, model.r, model.f1, model.count],
            #     feed_dict)
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # print("{}: step {}, loss {:g}".format(time_str, step, loss))

            # train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev/test set
            """
            feed_dict = {
              model.input_x: x_batch,
              model.input_y: y_batch,
              model.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("-----{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # if writer:
            #     writer.add_summary(summaries, step)


        # Generate batches
        batches = batch_iter(
            list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)


        # # continue train
        if FLAGS.continue_flag:
           checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
           saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
           saver.restore(sess, checkpoint_file)

        if FLAGS.test_mode:
            dev_step(test_x, test_y)
            sys.exit(0)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                # dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # dev_step(x_test, y_test, writer=test_summary_writer)
                # dev_step(x_dev, y_dev)
                dev_step(test_x, test_y)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))