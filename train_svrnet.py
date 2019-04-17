from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from datetime import datetime

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train SVRnet')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/data/svrnet-dataset',
    help='The path to the dataset directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='ifind',
    help='The name of the dataset to use. \
    Available names are: ifind | kingscollege | tum')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/SVRnet',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--params', type=str, default='/tmp/models/param.json',
    help='Auxiliary parameters file.')
# training parameters
ARGPARSER.add_argument(
    '--train_iter', type=int, default=200000,
    help='The number of epochs to train.')
ARGPARSER.add_argument(
    '--init_lr', type=float, default=1e-4,
    help='Initial Learning rate.')
ARGPARSER.add_argument(
    '--loss', type=str, default='PoseNet',
    help='The loss function used. \
    Available loss are: AP | PoseNet | SE3')
# snapshot parameters
ARGPARSER.add_argument(
    '--ckpt_steps', type=int, default=1000,
    help='Number of steps between checkpoint saves.')
# memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=36,
    help='The number of data points per batch.')
ARGPARSER.add_argument(
    '--memcap', type=float, default=1.0,
    help='Maximum fraction of memory to allocate per GPU.')
# data loading
ARGPARSER.add_argument(
    '--queue_threads', type=int, default=4,
    help='How many parallel threads to run for data queuing.')
ARGPARSER.add_argument(
    '--queue_buffer', type=int, default=1000,
    help='How many samples to queue up.')
# logging
ARGPARSER.add_argument(
    '--log_steps', type=int, default=10,
    help='Global steps between log output.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")
ARGPARSER.add_argument(
    '--initial_eval', default=False, action='store_true',
    help="Runs an evaluation before the first training iteration.")
# multi-gpu systems
ARGPARSER.add_argument(
    '--gpu', type=str, default='0',
    help='Specify default GPU to use.')


def _parse_function_ifind(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image' : tf.FixedLenFeature([], tf.string),
            'vec'   : tf.FixedLenFeature([], tf.string),
            'qt'    : tf.FixedLenFeature([], tf.string),
            'AP1'   : tf.FixedLenFeature([], tf.string),
            'AP2'   : tf.FixedLenFeature([], tf.string),
            'AP3'   : tf.FixedLenFeature([], tf.string)})

    image   = tf.reshape(tf.decode_raw(features['image'], tf.float32),[120,120,1]) * 255.
    vec     = tf.reshape(tf.decode_raw(features['vec'], tf.float32),[3])
    qt      = tf.reshape(tf.decode_raw(features['qt'], tf.float32),[4])
    AP1     = tf.reshape(tf.decode_raw(features['AP1'], tf.float32),[3])
    AP2     = tf.reshape(tf.decode_raw(features['AP2'], tf.float32),[3])
    AP3     = tf.reshape(tf.decode_raw(features['AP3'], tf.float32),[3])

    return image, vec, qt, AP1, AP2, AP3


def _parse_function_kingscollege(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image':    tf.FixedLenFeature([], tf.string),
            'vec':      tf.FixedLenFeature([], tf.string),
            'pose_q':   tf.FixedLenFeature([], tf.string),
            'pose_x':   tf.FixedLenFeature([], tf.string)
        })

    image =     tf.decode_raw(features['image'], tf.uint8)
    vec =       tf.decode_raw(features['vec'], tf.float32)
    pose_q =    tf.decode_raw(features['pose_q'], tf.float32)
    pose_x =    tf.decode_raw(features['pose_x'], tf.float32)

    image = tf.reshape(image, (270, 480, 3))
    vec.set_shape((3))
    pose_q.set_shape((4))
    pose_x.set_shape((3))

    return image, vec, pose_q, 0, pose_x, 0


def main(argv):

    if FLAGS.dataset == 'ifind':
        files = tf.data.Dataset.list_files(tf.gfile.Glob(FLAGS.data_dir + '/train/*'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        dataset = dataset.map(_parse_function_ifind)

    elif FLAGS.dataset == 'kingscollege':
        dataset = tf.data.TFRecordDataset(FLAGS.data_dir + '/dataset_train.tfrecords')
        dataset = dataset.map(_parse_function_kingscollege)

    else:
        print('Invalid Option:', FLAGS.dataset)
        raise SystemExit

    dataset = dataset.repeat()
    dataset = dataset.shuffle(FLAGS.queue_buffer)
    dataset = dataset.batch(FLAGS.batch_size)
    image, vec, qt, AP1, AP2, AP3 = dataset.make_one_shot_iterator().get_next()


    # Network Definition

    image = tf.image.resize_images(image, size=[224, 224])
    # tf.summary.image('input',image,max_outputs=30)

    if FLAGS.loss == 'PoseNet':

        y_pred, _ = inception.inception_v3(image, num_classes=7)
        quaternion_pred, translation_pred = tf.split(y_pred, [4, 3], axis=1)

        l1 = tf.nn.l2_loss(quaternion_pred - qt, name='loss/l1')
        l2 = tf.nn.l2_loss(translation_pred - AP2, name='loss/l2')

        tf.summary.scalar('PoseNet_loss/loss_quaternion', l1)
        tf.summary.scalar('PoseNet_loss/loss_translation', l2)

        loss = l1 * 500 + l2

    elif FLAGS.loss == 'AP':

        y_pred, _ = inception.inception_v3(image, num_classes=9)
        AP1_pred, AP2_pred, AP3_pred = tf.split(y_pred, 3, axis=1)

        l1 = tf.nn.l2_loss(AP1_pred - AP1, name='loss/l1')
        l2 = tf.nn.l2_loss(AP2_pred - AP2, name='loss/l2')
        l3 = tf.nn.l2_loss(AP3_pred - AP3, name='loss/l3')

        tf.summary.scalar('AnchorPoints_loss/AP1', l1)
        tf.summary.scalar('AnchorPoints_loss/AP2', l2)
        tf.summary.scalar('AnchorPoints_loss/AP3', l3)

        loss = l1 + l2 + l3

    elif FLAGS.loss == 'SE3':

        from se3_geodesic_loss import SE3GeodesicLoss

        SE3_DIM = 6

        # SE3 Training Weights for KingsCollege Dataset:
        se3_weights = np.ones(SE3_DIM) # Default
        # se3_weights = np.array([0.77848403, 0.6148858 , 0.12600519, 0.00018093, 0.00020279, 0.00082466])
        loss = SE3GeodesicLoss(se3_weights)

        y_pred, _ = inception.inception_v3(image, num_classes=SE3_DIM)
        y_pred.set_shape([FLAGS.batch_size,SE3_DIM])

        y_true = tf.concat((vec,AP2),axis=1)
        y_true.set_shape([FLAGS.batch_size,SE3_DIM])

        with tf.variable_scope('SE3_loss'):
            loss = loss.geodesic_loss(y_pred, y_true)

    else:
        print('Invalid Option:',FLAGS.loss)
        raise SystemExit

    tf.summary.scalar('loss', loss)


    # Optimiser
    train_op = tf.train.AdamOptimizer(FLAGS.init_lr).minimize(loss)

    # Session Configuration
    config = tf.ConfigProto(
        # log_device_placement=True,
        # allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        initializer = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

        sess.run(initializer)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.model_dir + '/train', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(FLAGS.train_iter):
                start_time = time.time()

                _, summary, loss_value = sess.run([train_op, merged, loss])

                train_writer.add_summary(summary, step)
                duration = time.time() - start_time
                if step % FLAGS.log_steps == 0:

                    examples_per_sec = FLAGS.batch_size / float(duration)
                    format_str = ('%s: step %d, loss = %.5f '
                                  '(%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, duration))

                if step % FLAGS.chkpt_steps == 0:
                    tf.logging.log(tf.logging.INFO, 'Saving Iteration: {}'.format(step))
                    saver.save(sess, FLAGS.model_dir + '/iter_{}.ckpt'.format(step))

        except tf.errors.OutOfRangeError:
            # End of dataset
            tf.logging.log(tf.logging.INFO, 'End of Training')

        except KeyboardInterrupt:
            tf.logging.log(tf.logging.INFO, 'Keyboard Interrupt!')

        finally:
            tf.logging.log(tf.logging.INFO, 'Stopping Threads')
            coord.request_stop()
            coord.join(threads)
            tf.logging.log(tf.logging.INFO, 'Saving iter: {}'.format(step))
            saver.save(sess, FLAGS.model_dir + '/iter_{}.ckpt'.format(step))


if __name__ == '__main__':

    print('Training Pose Estimation Model.')
    FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
    print('FLAGS:', FLAGS)
    print('UNPARSED_ARGV:', UNPARSED_ARGV)

    # Set verbosity
    if FLAGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    # using the Winograd non-fused algorithms provides a small performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED_ARGV)
