from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imageio
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception
from geomstats.invariant_metric import InvariantMetric
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
from geomstats.special_euclidean_group import SpecialEuclideanGroup

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Evaluate SVRnet KingsCollege')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/data/kingscollege-dataset',
    help='The path to the dataset directory.')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/SVRnet',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--loss', type=str, default='PoseNet',
    help='The loss function used. \
    Available loss are: PoseNet | SE3')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")


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

    return image, vec, pose_q, pose_x


def main(argv):

    # TF Record
    dataset = tf.data.TFRecordDataset(FLAGS.data_dir + '/dataset_test.tfrecords')
    dataset = dataset.map(_parse_function_kingscollege)
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(FLAGS.queue_buffer)
    dataset = dataset.batch(1)
    image, vec, pose_q, pose_x = dataset.make_one_shot_iterator().get_next()

    # Network Definition
    image_resized = tf.image.resize_images(image, size=[224, 224])

    if FLAGS.loss == 'PoseNet':

        y_pred, _ = inception.inception_v3(image_resized, num_classes=7, is_training=False)
        quaternion_pred, translation_pred = tf.split(y_pred, [4, 3], axis=1)

        sess = tf.Session()

        ckpt_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.train.Saver().restore(sess, ckpt_file)
        print('restoring parameters from', ckpt_file)

        i = 0

        try:

            while True:
                _image, _quaternion_true, _translation_true, _quaternion_pred, _translation_pred = \
                    sess.run([image, pose_q, pose_x, quaternion_pred, translation_pred])

                # Compute Individual Sample Error
                q1 = _quaternion_true / np.linalg.norm(_quaternion_true)
                q2 = _quaternion_pred / np.linalg.norm(_quaternion_pred)
                d = abs(np.sum(np.multiply(q1, q2)))
                theta = 2. * np.arccos(d) * 180. / np.pi
                error_x = np.linalg.norm(_translation_true - _translation_pred)

                print('Iteration:', i, 'Error XYZ (m):', error_x, 'Error Q (degrees):', theta)
                i = i + 1

        except tf.errors.OutOfRangeError:
            print('End of Test Data')


    elif FLAGS.loss == 'SE3':

        y_pred, _ = inception.inception_v3(image_resized, num_classes=6, is_training=False)

        sess = tf.Session()

        ckpt_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.train.Saver().restore(sess, ckpt_file)
        print('restoring parameters from', ckpt_file)

        SO3_GROUP = SpecialOrthogonalGroup(3)
        SE3_GROUP = SpecialEuclideanGroup(3)
        metric = InvariantMetric(group=SE3_GROUP, inner_product_mat_at_identity=np.eye(6), left_or_right='left')

        i = 0

        _y_pred_i = []
        _y_true_i = []
        _se3_err_i = []

        try:

            while True:
                _image, _rvec, _qvec, _tvec, _y_pred = \
                    sess.run([image, vec, pose_q, pose_x, y_pred])

                _quaternion_true = _qvec
                _quaternion_pred = SO3_GROUP.quaternion_from_rotation_vector(_y_pred[0,:3])[0]

                # Compute Individual Sample Error
                q1 = _quaternion_true / np.linalg.norm(_quaternion_true)
                q2 = _quaternion_pred / np.linalg.norm(_quaternion_pred)
                d = abs(np.sum(np.multiply(q1, q2)))
                theta = 2. * np.arccos(d) * 180. / np.pi
                error_x = np.linalg.norm(_tvec - _y_pred[0,3:])

                # SE3 compute
                _y_true = np.concatenate((_rvec,_tvec),axis=-1)
                se3_dist = metric.squared_dist(_y_pred, _y_true)[0]

                _y_pred_i.append(_y_pred)
                _y_true_i.append(_y_true)
                _se3_err_i.append(SE3_GROUP.compose(SE3_GROUP.inverse(_y_true),_y_pred))

                print('Iteration:', i, 'Error XYZ (m):', error_x, 'Error Q (degrees):', theta, 'SE3 dist:', se3_dist)
                i = i + 1


        except tf.errors.OutOfRangeError:
            print('End of Test Data')

        # Calculate SE3 Error Weights
        err_vec = np.vstack(_se3_err_i)
        err_weights = np.diag(np.linalg.inv(np.cov(err_vec.T)))
        err_weights = err_weights / np.linalg.norm(err_weights)
        print(err_weights)

    else:
        print('Invalid Option:',FLAGS.loss)
        raise SystemExit




if __name__ == '__main__':

    print('Evaluate SVRnet Model on KingsCollege data')
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

    # using the Winograd non-fused algorithms provides a small performance boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    tf.app.run(main=main, argv=[sys.argv[0]] + UNPARSED_ARGV)
