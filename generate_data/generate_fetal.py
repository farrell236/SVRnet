from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import imageio
import numpy as np
import SimpleITK as sitk
import transformations as T
import tensorflow as tf

import sys
sys.path.append('/vol/medic01/users/bh1511/_build/geomstats-farrell/')
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

###############################################################################
# Tensorflow feature wrapper

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


###############################################################################
# Dataset Locations

NIFTI_ROOT  = '/vol/medic01/users/bh1511/DATA_RAW/AliceBrainReconsAligned/test/'
SAVE_DIR    = ''

n_rotations = 2000
n_offsets   = 16
max_z       = 40

###############################################################################
# Data Generation

SO3_GROUP   = SpecialOrthogonalGroup(3)

database    = glob.glob(NIFTI_ROOT+'/*.nii.gz')

for fetal_brain in database:

    print('Parsing:', fetal_brain)

    fixed_image_sitk_tmp    = sitk.ReadImage(fetal_brain, sitk.sitkFloat32)
    fixed_image_sitk        = sitk.GetImageFromArray(sitk.GetArrayFromImage(fixed_image_sitk_tmp))
    fixed_image_sitk        = sitk.RescaleIntensity(fixed_image_sitk, 0, 1)

    writer      = tf.python_io.TFRecordWriter(SAVE_DIR +
        os.path.basename(fetal_brain).replace('.nii.gz','.tfrecord'))

    rotations   = np.pi * (np.random.rand(n_rotations, 3) - 0.5)

    for rotation in rotations:

        R           = T.euler_matrix(*rotation)[:3,:3]
        Rinv        = np.linalg.inv(R)

        size        = fixed_image_sitk.GetSize()
        spacing     = fixed_image_sitk.GetSpacing()
        new_origin  = (60,60,60) - R.dot(np.array(size)/2)

        fixed_image_sitk.SetOrigin(new_origin)
        fixed_image_sitk.SetDirection(R.flatten())

        # resample filter
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetOutputDirection((1,0,0,0,1,0,0,0,1)) #Identity
        resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
        resampleFilter.SetOutputSpacing(spacing)
        resampleFilter.SetOutputOrigin((0,0,0))
        resampleFilter.SetDefaultPixelValue(0)
        resampleFilter.SetSize((120,120,120))
        resampleFilter.SetNumberOfThreads(20)

        # transform the image
        moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)
        npimage = sitk.GetArrayFromImage(moving_image_sitk) 

        offsets = np.random.randint(60-max_z, 60+max_z, size=n_offsets)

        for offset in offsets:

            img     = npimage[offset,...]
            vec     = SO3_GROUP.rotation_vector_from_matrix(Rinv)[0]
            qt      = SO3_GROUP.quaternion_from_matrix(Rinv)[0] 
            AP1     = Rinv.dot((-60.,-60.,(offset-60.))) / 60. # Left 
            AP2     = Rinv.dot((  0.,  0.,(offset-60.))) / 60. # Origin 
            AP3     = Rinv.dot(( 60.,-60.,(offset-60.))) / 60. # Right

            img_raw = img.astype('float32').tostring()
            vec_raw = vec.astype('float32').tostring()
            qt_raw  = qt.astype('float32').tostring()
            AP1_raw = AP1.astype('float32').tostring()
            AP2_raw = AP2.astype('float32').tostring()
            AP3_raw = AP3.astype('float32').tostring()
            
            example = tf.train.Example(features=tf.train.Features(feature={
                'image':    _bytes_feature(img_raw),
                'vec':      _bytes_feature(vec_raw),
                'qt':       _bytes_feature(qt_raw),
                'AP1':      _bytes_feature(AP1_raw),
                'AP2':      _bytes_feature(AP2_raw),
                'AP3':      _bytes_feature(AP3_raw)}))

            writer.write(example.SerializeToString())

    writer.close()

