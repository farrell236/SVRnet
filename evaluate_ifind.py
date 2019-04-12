from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imageio
import os
import sys

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
from evaluate_quality_img import (
    calc_psnr, calc_ssim, calc_mse, calc_correlation)

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Evaluate SVRnet iFIND')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/data/ifind-dataset',
    help='The path to the dataset directory.')
ARGPARSER.add_argument(
    '--scan_dir', type=str, default='/tmp/data/ifind-dataset',
    help='The path to the nifti volume directory.')
ARGPARSER.add_argument(
    '--subject_id', type=str, default='recon-ifind00080',
    help='Subject ID to evaluate')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/SVRnet',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--loss', type=str, default='PoseNet',
    help='The loss function used. \
    Available loss are: AP | PoseNet | SE3')
ARGPARSER.add_argument(
    '--n_iter', type=int, default=100,
    help='The number of epochs to train.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")


def matrix_from_anchor_points(AP1, AP2, AP3):
    v1 = AP3 - AP1
    v2 = AP2 - AP1
    n1 = np.cross(v1, v2)
    n2 = np.cross(n1, v1)

    v1_norm = v1 / np.linalg.norm(v1)  # x
    n2_norm = n2 / np.linalg.norm(n2)  # y
    n1_norm = n1 / np.linalg.norm(n1)  # z

    # SimpleITK does not like this...
    # R_recon = np.vstack((v1_norm, n2_norm, n1_norm)).T

    R_recon = np.zeros([3, 3])
    R_recon[0, 0] = v1_norm[0]
    R_recon[0, 1] = n2_norm[0]
    R_recon[0, 2] = n1_norm[0]
    R_recon[1, 0] = v1_norm[1]
    R_recon[1, 1] = n2_norm[1]
    R_recon[1, 2] = n1_norm[1]
    R_recon[2, 0] = v1_norm[2]
    R_recon[2, 1] = n2_norm[2]
    R_recon[2, 2] = n1_norm[2]

    return R_recon


def resample_sitk(fixed_image_sitk, rx, tx):

    # rx is the rotation of the plane, R is the rotation of the brain
    R = np.linalg.inv(rx)

    size        = fixed_image_sitk.GetSize()
    spacing     = fixed_image_sitk.GetSpacing()

    new_origin  = (60, 60, 60) - R.dot(np.array(size) / 2) - R.dot(tx)

    fixed_image_sitk.SetOrigin(new_origin)
    fixed_image_sitk.SetDirection(np.array(R.flatten()))

    # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Identity
    resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin((0, 0, 0))
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize((120, 120, 120))

    # transform the image
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)

    npimage = sitk.GetArrayFromImage(moving_image_sitk)[60,...]

    return npimage


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

    image   = tf.reshape(tf.decode_raw(features['image'], tf.float32),[120,120,1]) #* 255.
    vec     = tf.reshape(tf.decode_raw(features['vec'], tf.float32),[3])
    qt      = tf.reshape(tf.decode_raw(features['qt'], tf.float32),[4])
    AP1     = tf.reshape(tf.decode_raw(features['AP1'], tf.float32),[3])
    AP2     = tf.reshape(tf.decode_raw(features['AP2'], tf.float32),[3])
    AP3     = tf.reshape(tf.decode_raw(features['AP3'], tf.float32),[3])

    return image, vec, qt, AP1, AP2, AP3


def main(argv):

    # TF Record
    datafiles = FLAGS.data_dir + '/test/' + FLAGS.subject_id + '.tfrecord'
    dataset = tf.data.TFRecordDataset(datafiles)
    dataset = dataset.map(_parse_function_ifind)
    # dataset = dataset.repeat()
    # dataset = dataset.shuffle(FLAGS.queue_buffer)
    dataset = dataset.batch(1)
    image, vec, qt, AP1, AP2, AP3 = dataset.make_one_shot_iterator().get_next()

    # Nifti Volume
    subject_path = FLAGS.scan_dir + '/test/' + FLAGS.subject_id + '.nii.gz'
    fixed_image_sitk_tmp    = sitk.ReadImage(subject_path, sitk.sitkFloat32)
    fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
    fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1) # * 255.

    # Network Definition
    image_resized = tf.image.resize_images(image, size=[224, 224])

    if FLAGS.loss == 'PoseNet':

        y_pred, _ = inception.inception_v3(image_resized, num_classes=7, is_training=False)
        quaternion_pred, translation_pred = tf.split(y_pred, [4, 3], axis=1)

        sess = tf.Session()

        ckpt_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.train.Saver().restore(sess, ckpt_file)
        print('restoring parameters from', ckpt_file)

        SO3_GROUP = SpecialOrthogonalGroup(3)

        for i in range(FLAGS.n_iter):

            _image, _quaternion_true, _translation_true, _quaternion_pred, _translation_pred = \
                sess.run([image, qt, AP2, quaternion_pred, translation_pred])

            rx = SO3_GROUP.matrix_from_quaternion(_quaternion_pred)[0]
            tx = _translation_pred[0] * 60.

            image_true = np.squeeze(_image)
            image_pred = resample_sitk(fixed_image_sitk, rx, tx)

            imageio.imsave('imgdump/image_{}_true.png'.format(i),_image[0,...])
            imageio.imsave('imgdump/image_{}_pred.png'.format(i),image_pred)

            calc_psnr(image_pred, image_true)
            calc_mse(image_pred, image_true)
            calc_ssim(image_pred, image_true)
            calc_correlation(image_pred, image_true)

    elif FLAGS.loss == 'AP':

        y_pred, _ = inception.inception_v3(image_resized, num_classes=9, is_training=False)
        AP1_pred, AP2_pred, AP3_pred = tf.split(y_pred, 3, axis=1)

        sess = tf.Session()

        ckpt_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.train.Saver().restore(sess, ckpt_file)
        print('restoring parameters from', ckpt_file)

        for i in range(FLAGS.n_iter):

            _image, _AP1, _AP2, _AP3, _AP1_pred, _AP2_pred, _AP3_pred = \
                sess.run([image, AP1, AP2, AP3, AP1_pred, AP2_pred, AP3_pred])

            dist_ap1 = np.linalg.norm(_AP1 - _AP1_pred)
            dist_ap2 = np.linalg.norm(_AP2 - _AP2_pred)
            dist_ap3 = np.linalg.norm(_AP3 - _AP3_pred)

            rx = matrix_from_anchor_points(_AP1_pred[0], _AP2_pred[0], _AP3_pred[0])
            tx = _AP2_pred[0] * 60.

            image_true = np.squeeze(_image)
            image_pred = resample_sitk(fixed_image_sitk, rx, tx)

            imageio.imsave('imgdump/image_{}_true.png'.format(i),_image[0,...])
            imageio.imsave('imgdump/image_{}_pred.png'.format(i),image_pred)

            calc_psnr(image_pred, image_true)
            calc_mse(image_pred, image_true)
            calc_ssim(image_pred, image_true)
            calc_correlation(image_pred, image_true)


    elif FLAGS.loss == 'SE3':

        y_pred, _ = inception.inception_v3(image_resized, num_classes=6, is_training=False)

        sess = tf.Session()

        ckpt_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        tf.train.Saver().restore(sess, ckpt_file)
        print('restoring parameters from', ckpt_file)

        SO3_GROUP = SpecialOrthogonalGroup(3)

        for i in range(FLAGS.n_iter):

            _image, _rvec, _tvec, _y_pred = \
                sess.run([image, vec, AP2, y_pred])

            rx = SO3_GROUP.matrix_from_rotation_vector(_y_pred[0,:3])[0]
            tx = _y_pred[0,3:] * 60.

            image_true = np.squeeze(_image)
            image_pred = resample_sitk(fixed_image_sitk, rx, tx)

            imageio.imsave('imgdump/image_{}_true.png'.format(i),_image[0,...])
            imageio.imsave('imgdump/image_{}_pred.png'.format(i),image_pred)

            calc_psnr(image_pred, image_true)
            calc_mse(image_pred, image_true)
            calc_ssim(image_pred, image_true)
            calc_correlation(image_pred, image_true)

    else:
        print('Invalid Option:',FLAGS.loss)
        raise SystemExit


if __name__ == '__main__':

    print('Evaluate SVRnet Model on iFIND data')
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
