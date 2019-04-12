# ================================================================================
# Copyright (c) 2018 Benjamin Hou (bh1511@imperial.ac.uk)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ================================================================================

import numpy as np
import tensorflow as tf


from geomstats.invariant_metric import InvariantMetric
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from tensorflow.python.framework import ops


SE3_DIM = 6
N = 3

class SE3GeodesicLoss(object):
    """
    Geodesic Loss on the Special Euclidean Group SE(3), of 3D rotations 
    and translations, computed as the square geodesic distance with respect 
    to a left-invariant Riemannian metric.
    """
    def __init__(self, weight, op_name='SE3GeodesicLoss'):

        assert weight.shape != SE3_DIM, 'Weight vector must be of shape [6]'

        self.op_name = op_name
        self.weight = weight
        self.SE3_GROUP = SpecialEuclideanGroup(N)
        self.metric = InvariantMetric( 
            group=self.SE3_GROUP, 
            inner_product_mat_at_identity=np.eye(SE3_DIM) * self.weight, 
            left_or_right='left')

    # Python Custom Op Tensorflow Wrapper
    def py_func(self, func, inp, Tout, stateful=True, name=None, grad=None):
        """
        PyFunc defined as given by Tensorflow

        :param func:        Custom Function
        :param inp:         Function Inputs
        :param Tout:        Ouput Type of out Custom Function
        :param stateful:    Calculate Gradients when stateful is True
        :param name:        Name of the PyFunction
        :param grad:        Custom Gradient Function
        :return:
        """
        # Generate Random Gradient name to avoid conflicts with inbuilt names
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 2**32-1))

        # Register Tensorflow Gradient
        tf.RegisterGradient(rnd_name)(grad)

        # Get current graph
        g = tf.get_default_graph()

        # Add gradient override map
        with g.gradient_override_map({
                'PyFunc': rnd_name,
                'PyFuncStateless': rnd_name}):
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    # Python Custom Op
    def geodesic_loss(self, y_pred, y_true, name=None):
        """
        Custom Function which defines pyfunc and gradient override
        :param x:       y_pred - predicted se(3) pose
        :param y:       y_true - ground truth se(3) pose
        :param name:    Function name
        :return:        dist - geodesic distance between y_pred and y_true
        """
        with ops.name_scope(name, self.op_name, [y_pred, y_true]) as name:
            """
            Our pyfunc op accepts 2 input parameters and returns 1 outputs
            Input Parameters:   y_pred, y_true
            Output Parameters:  geodesic distance
            """
            dist, grad = self.py_func(self.riemannian_dist_grad,
                                      [y_pred, y_true],
                                      [tf.float32, tf.float32],
                                      name=name,
                                      grad=self.riemannian_grad_op)
            return dist

    # Geodesic Loss Core Function
    def riemannian_dist_grad(self, y_pred, y_true):
        """
        Geodesic Loss Core Function

        :param y_pred: y_pred
        :param y_true: y_true
        :return: dist, grad
        """
        # Geodesic Distance
        sq_geodesic_dist = self.metric.squared_dist(y_pred, y_true)
        batch_loss = np.sum(sq_geodesic_dist).astype('float32')

        # Computation of Riemannian Gradient
        tangent_vec = self.metric.log(base_point=y_pred, point=y_true)

        grad_point = - 2. * tangent_vec

        inner_prod_mat = self.metric.inner_product_matrix(base_point=y_pred)

        riemannian_grad = np.einsum('ijk,ik->ij', inner_prod_mat, grad_point)

        sqrt_weight = np.sqrt(self.weight)
        riemannian_grad = np.multiply(riemannian_grad, sqrt_weight).astype('float32')

        return batch_loss, riemannian_grad

    # Geodesic Loss Gradient Function
    def riemannian_grad_op(self, op, grads, grad_glob):
        """
        Geodesic Loss Gradient Function

        :param op:          Operation - operation.inputs  = [y_pred, y_true],
                                        operation.outputs = [dist, grad]
        :param grads:       Gradients for equation prime
        :param grad_glob:   No real use of it, but the gradient function
                            parameter size should match op.inputs
        :return:            grads * d/d_y_pred , vector of ones
        """
        # Only gradient w.r.t. y_pred is returned.
        return grads * op.outputs[1], tf.ones_like(op.outputs[1])
