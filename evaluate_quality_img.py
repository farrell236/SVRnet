# library path
import os, sys

# lib_path = os.path.abspath(
#     '/vol/biomedic/users/aa16914/software/SimpleITK/SimpleITK-build/SimpleITK-build/Wrapping/Python/')
# sys.path.insert(1, lib_path)

import SimpleITK   as sitk
import numpy       as np
import math        as mt

# ssim
from scipy.ndimage import uniform_filter, gaussian_filter
from numpy.lib.arraypad import _validate_lengths

# multi-processing
import multiprocessing

num_cores = multiprocessing.cpu_count()


###############################################################
# dtype_range = {np.bool_: (False, True),
#                np.bool8: (False, True),
#                np.uint8: (0, 255),
#                np.uint16: (0, 65535),
#                np.uint32: (0, 2**32 - 1),
#                np.uint64: (0, 2**64 - 1),
#                np.int8: (-128, 127),
#                np.int16: (-32768, 32767),
#                np.int32: (-2**31, 2**31 - 1),
#                np.int64: (-2**63, 2**63 - 1),
#                np.float16: (-1, 1),
#                np.float32: (-1, 1),
#                np.float64: (-1, 1)}
#
###############################################################
def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    if im1.dtype != float_type:
        im1 = im1.astype(float_type)
    if im2.dtype != float_type:
        im2 = im2.astype(float_type)
    return im1, im2


###############################################################
def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.
    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.
    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


###############################################################
def register(moving_image, fixed_image):
    """Resample the target image to the reference image.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    resampled : sitk
        the resampled target image
    """
    transfromDomainMeshSize = [10] * moving_image.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_image,
                                          transfromDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetShrinkFactorsPerLevel([6, 2, 1])
    R.SetSmoothingSigmasPerLevel([6, 2, 1])
    R.SetNumberOfThreads(num_cores)

    outTx = R.Execute(fixed_image, moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    resampler.SetNumberOfThreads(num_cores)

    return resampler.Execute(moving_image)


###############################################################
def register_rigid(moving_image, fixed_image):
    """Resample the target image to the reference image.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    resampled : sitk
        the resampled target image
    """

    numberOfBins = 24
    samplingPercentage = 0.10

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage)
    R.SetMetricSamplingStrategy(R.RANDOM)

    # -------------------------------------------------------------------------------------------------------------------------------
    # dsiplacement
    displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,
                                                varianceForTotalField=1.5)

    R.SetMovingInitialTransform(outTx)
    R.SetInitialTransform(displacementTx, inPlace=True)

    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()
    R.MetricUseFixedImageGradientFilterOff()

    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])

    R.SetOptimizerScalesFromPhysicalShift()
    R.SetOptimizerAsGradientDescent(learningRate=1,
                                    numberOfIterations=300,
                                    estimateLearningRate=R.EachIteration)

    outTx.AddTransform(R.Execute(fixed, moving))
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    R.SetNumberOfThreads(num_cores)
    outTx = R.Execute(fixed_image, moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    resampler.SetNumberOfThreads(num_cores)

    return resampler.Execute(moving_image)


###############################################################
def resample(tar_img, ref_img):
    """Resample the target image to the reference image.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    resampled : sitk
        the resampled target image
    """

    resizeFilter = sitk.ResampleImageFilter()
    resizeFilter.SetNumberOfThreads(num_cores)
    resizeFilter.SetReferenceImage(ref_img)

    return resizeFilter.Execute(tar_img)


###############################################################
def calc_correlation(tar_img, ref_img):
    """Resample the target image to the reference image.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    cross-correlation : float
        Cross-correlation of two images
    """

    tar_vol = tar_img
    ref_vol = ref_img

    num = np.sum((tar_vol - tar_vol.mean()) * (ref_vol - ref_vol.mean()))
    den = np.sqrt(np.sum(np.square(tar_vol - tar_vol.mean())) * np.sum(np.square(ref_vol - ref_vol.mean())))

    return num / den


###############################################################
def calc_mse(tar_img, ref_img):
    """Compute the mean-squared error between two images.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    """

    tar_vol = tar_img
    ref_vol = ref_img

    return np.mean(np.square(ref_vol - tar_vol), dtype=np.float64)


###############################################################
def calc_nrmse(tar_img, ref_img, norm_type='Euclidean'):
    """Compute the normalized root mean-squared error (NRMSE) between two images.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.
    norm_type : {'Euclidean', 'min-max', 'mean'}
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'Euclidean' : normalize by the Euclidean norm of ``im_true``.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``.
    Returns
    -------
    nrmse : float
        The NRMSE metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """

    tar_vol = tar_img
    ref_vol = ref_img

    norm_type = norm_type.lower()
    if norm_type == 'euclidean':
        denom = np.sqrt(np.mean((ref_vol * ref_vol), dtype=np.float64))
    elif norm_type == 'min-max':
        denom = ref_vol.max() - ref_vol.min()
    elif norm_type == 'mean':
        denom = ref_vol.mean()
    else:
        raise ValueError("Unsupported norm_type")

    return np.sqrt(calc_mse(ref_img, tar_img)) / denom


###############################################################
def calc_psnr(tar_img, ref_img):
    """ Compute the peak signal to noise ratio (PSNR) for an image.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.

    Returns
    -------
    psnr : float
        The PSNR metric.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """

    tar_vol = tar_img
    ref_vol = ref_img

    ref_vol, tar_vol = _as_floats(ref_vol, tar_vol)

    err = calc_mse(ref_img, tar_img)

    return 10 * np.log10((256 ** 2) / err)


###############################################################
def calc_ssim(tar_img, ref_img, win_size=None, gradient=False, gaussian_weights=False, full=False, **kwargs):
    """Compute the mean structural similarity index between two images.
    Parameters
    ----------
    tar_img : sitk
        Test image.
    ref_img : sitk
        Ground-truth image.
    win_size : int or None
        The side-length of the sliding window used in comparison.  Must be an
        odd value.  If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, return the full structural similarity image instead of the
        mean value.
    Other Parameters
    ----------------
    use_sample_covariance : bool
        if True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1]_)
    K2 : float
        algorithm parameter, K2 (small constant, see [1]_)
    sigma : float
        sigma for the Gaussian when `gaussian_weights` is True.
    Returns
    -------
    mssim : float
        The mean structural similarity over the image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.
    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.
    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1.1.11.2477
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
       DOI:10.1007/s10043-009-0119-z
    """

    tar_vol = tar_img
    ref_vol = ref_img

    data_range = 256.0

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(tar_vol.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    ndim = tar_vol.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    tar_vol = tar_vol.astype(np.float64)
    ref_vol = ref_vol.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(tar_vol, **filter_args)
    uy = filter_func(ref_vol, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(tar_vol * tar_vol, **filter_args)
    uyy = filter_func(ref_vol * ref_vol, **filter_args)
    uxy = filter_func(tar_vol * ref_vol, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * tar_vol
        grad += filter_func(-S / B2, **filter_args) * ref_vol
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / tar_vol.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


###############################################################
###############################################################
if __name__ == "__main__":
    ref_img = sitk.ReadImage('recon_gt.nii.gz')
    # tar_img = sitk.ReadImage('recon_cnn.nii.gz')
    # tar_img = sitk.ReadImage('recon_bspline.nii.gz')
    # tar_img = sitk.ReadImage('recon_linear.nii.gz')
    tar_img = sitk.ReadImage('recon_downsampled.nii.gz')

    # crop
    ref_img = sitk.RegionOfInterest(ref_img, size=[ref_img.GetSize()[0] - 20, ref_img.GetSize()[1] - 20,
                                                   ref_img.GetSize()[2] - 20], index=[10, 10, 10])
    tar_img = sitk.RegionOfInterest(tar_img, size=[tar_img.GetSize()[0] - 20, tar_img.GetSize()[1] - 20,
                                                   tar_img.GetSize()[2] - 20], index=[10, 10, 10])

    # resample to reference image
    tar_img = resample(tar_img=tar_img, ref_img=ref_img)

    # register
    # tar_img = register(moving_image=tar_img, fixed_image=ref_img)

    # calculate psnr
    print(calc_psnr(tar_img=tar_img, ref_img=ref_img))

    # calculate cross-coreelation
    print(calc_correlation(tar_img=tar_img, ref_img=ref_img))

    # calculate ssim
    ssim, ssim_vol = calc_ssim(tar_img=tar_img, ref_img=ref_img, full=True)
    dssim_vol = (1 - ssim_vol) / 2.
    print(ssim)

    dssim_img = sitk.GetImageFromArray(dssim_vol)
    dssim_img.CopyInformation(tar_img)
    sitk.WriteImage(dssim_img, 'dssim.nii.gz')
