"""
Augmenters that are based on applying convolution kernels to images.

Do not import directly from this file, as the categorization is not final.
Use instead ::

    from imgaug import augmenters as iaa

and then e.g. ::

    seq = iaa.Sequential([
        iaa.Sharpen((0.0, 1.0)),
        iaa.Emboss((0.0, 1.0))
    ])

List of augmenters:

    * Convolve
    * Sharpen
    * Emboss
    * EdgeDetect
    * DirectedEdgeDetect

For MotionBlur, see ``blur.py``.

"""
from __future__ import print_function, division, absolute_import

import types
import itertools

import numpy as np
import cv2
import six.moves as sm

from . import meta
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


# TODO allow 3d matrices as input (not only 2D)
class Convolve(meta.Augmenter):
    """
    Apply a convolution to input images.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no (1)
        * ``uint64``: no (2)
        * ``int8``: yes; tested (3)
        * ``int16``: yes; tested
        * ``int32``: no (2)
        * ``int64``: no (2)
        * ``float16``: yes; tested (4)
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no (1)
        * ``bool``: yes; tested (4)

        - (1) rejected by ``cv2.filter2D()``.
        - (2) causes error: cv2.error: OpenCV(3.4.2) (...)/filter.cpp:4487:
              error: (-213:The function/feature is not implemented)
              Unsupported combination of source format (=1), and destination
              format (=1) in function 'getLinearFilter'.
        - (3) mapped internally to ``int16``.
        - (4) mapped internally to ``float32``.

    Parameters
    ----------
    matrix : None or (H, W) ndarray or imgaug.parameters.StochasticParameter or callable, optional
        The weight matrix of the convolution kernel to apply.

            * If ``None``, the input images will not be changed.
            * If a 2D numpy array, that array will always be used for all
              images and channels as the kernel.
            * If a callable, that method will be called for each image
              via ``parameter(image, C, random_state)``. The function must
              either return a list of ``C`` matrices (i.e. one per channel)
              or a 2D numpy array (will be used for all channels) or a
              3D ``HxWxC`` numpy array. If a list is returned, each entry may
              be ``None``, which will result in no changes to the respective
              channel.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> matrix = np.array([[0, -1, 0],
    >>>                    [-1, 4, -1],
    >>>                    [0, -1, 0]])
    >>> aug = iaa.Convolve(matrix=matrix)

    Convolves all input images with the kernel shown in the ``matrix``
    variable.

    >>> def gen_matrix(image, nb_channels, random_state):
    >>>     matrix_A = np.array([[0, -1, 0],
    >>>                          [-1, 4, -1],
    >>>                          [0, -1, 0]])
    >>>     matrix_B = np.array([[0, 1, 0],
    >>>                          [1, -4, 1],
    >>>                          [0, 1, 0]])
    >>>     if image.shape[0] % 2 == 0:
    >>>         return [matrix_A] * nb_channels
    >>>     else:
    >>>         return [matrix_B] * nb_channels
    >>> aug = iaa.Convolve(matrix=gen_matrix)

    Convolves images that have an even height with matrix A and images
    having an odd height with matrix B.

    """

    def __init__(self, matrix=None,
                 name=None, deterministic=False, random_state=None):
        super(Convolve, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        if matrix is None:
            self.matrix = None
            self.matrix_type = "None"
        elif ia.is_np_array(matrix):
            ia.do_assert(
                matrix.ndim == 2,
                "Expected convolution matrix to have exactly two dimensions, "
                "got %d (shape %s)." % (
                    matrix.ndim, matrix.shape))
            self.matrix = matrix
            self.matrix_type = "constant"
        elif isinstance(matrix, types.FunctionType):
            self.matrix = matrix
            self.matrix_type = "function"
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got %s." % (
                    type(matrix),))

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32", "float64"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)
        rss = ia.derive_random_states(random_state, len(images))

        for i, image in enumerate(images):
            _height, _width, nb_channels = images[i].shape

            input_dtype = image.dtype
            if image.dtype.type in [np.bool_, np.float16]:
                image = image.astype(np.float32, copy=False)
            elif image.dtype.type == np.int8:
                image = image.astype(np.int16, copy=False)

            if self.matrix_type == "None":
                matrices = [None] * nb_channels
            elif self.matrix_type == "constant":
                matrices = [self.matrix] * nb_channels
            elif self.matrix_type == "function":
                matrices = self.matrix(images[i], nb_channels, rss[i])
                if ia.is_np_array(matrices) and matrices.ndim == 2:
                    matrices = np.tile(
                        matrices[..., np.newaxis],
                        (1, 1, nb_channels))

                is_valid_list = (isinstance(matrices, list)
                                 and len(matrices) == nb_channels)
                is_valid_array = (ia.is_np_array(matrices)
                                  and matrices.ndim == 3
                                  and matrices.shape[2] == nb_channels)
                ia.do_assert(
                    is_valid_list or is_valid_array,
                    "Callable provided to Convole must return either a "
                    "list of 2D matrices (one per image channel) "
                    "or a 2D numpy array "
                    "or a 3D numpy array where the last dimension's size "
                    "matches the number of image channels. "
                    "Got type %s." % (type(matrices),)
                )

                if ia.is_np_array(matrices):
                    # Shape of matrices is currently (H, W, C), but in the
                    # loop below we need the first axis to be the channel
                    # index to unify handling of lists of arrays and arrays.
                    # So we move the channel axis here to the start.
                    matrices = matrices.transpose((2, 0, 1))
            else:
                raise Exception("Invalid matrix type")

            image_aug = image
            for channel in sm.xrange(nb_channels):
                if matrices[channel] is not None:
                    # ndimage.convolve caused problems here cv2.filter2D()
                    # always returns same output dtype as input dtype
                    image_aug[..., channel] = cv2.filter2D(
                        image_aug[..., channel],
                        -1,
                        matrices[channel]
                    )

            if input_dtype == np.bool_:
                image_aug = image_aug > 0.5
            elif input_dtype in [np.int8, np.float16]:
                image_aug = iadt.restore_dtypes_(image_aug, input_dtype)

            images[i] = image_aug

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        # TODO this can fail for some matrices, e.g. [[0, 0, 1]]
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        # TODO this can fail for some matrices, e.g. [[0, 0, 1]]
        return keypoints_on_images

    def get_parameters(self):
        return [self.matrix, self.matrix_type]


def Sharpen(alpha=0, lightness=1,
            name=None, deterministic=False, random_state=None):
    """
    Sharpen images and alpha-blend the result with the original input images.

    dtype support::

        See ``imgaug.augmenters.convolutional.Convolve``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the sharpened image. At ``0.0``, only the original
        image is visible, at ``1.0`` only its sharpened version is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    lightness : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Lightness/brightness of the sharped image.
        Sane values are somewhere in the interval ``[0.5, 2.0]``.
        The value ``0.0`` results in an edge map. Values higher than ``1.0``
        create bright images. Default value is ``1.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Sharpen(alpha=(0.0, 1.0))

    Sharpens input images and blends the sharpened image with the input image
    using a random blending factor between ``0%`` and ``100%`` (uniformly
    sampled).

    >>> aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

    Sharpens input images with a variable `lightness` sampled uniformly from
    the interval ``[0.75, 2.0]`` and with a fully random blending factor
    (as in the above example).

    """
    alpha_param = iap.handle_continuous_param(
        alpha, "alpha",
        value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
    lightness_param = iap.handle_continuous_param(
        lightness, "lightness",
        value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        lightness_sample = lightness_param.draw_sample(
            random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1, -1, -1],
            [-1, 8+lightness_sample, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Convolve(
        create_matrices,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


def Emboss(alpha=0, strength=1,
           name=None, deterministic=False, random_state=None):
    """
    Emboss images and alpha-blend the result with the original input images.

    The embossed version pronounces highlights and shadows,
    letting the image look as if it was recreated on a metal plate ("embossed").

    dtype support::

        See ``imgaug.augmenters.convolutional.Convolve``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the embossed image. At ``0.0``, only the original
        image is visible, at ``1.0`` only its embossed version is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    strength : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Parameter that controls the strength of the embossing.
        Sane values are somewhere in the interval ``[0.0, 2.0]`` with ``1.0``
        being the standard embossing effect. Default value is ``1.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))

    Emboss an image with a strength sampled uniformly from the interval
    ``[0.5, 1.5]`` and alpha-blend the result with the original input image
    using a random blending factor between ``0%`` and ``100%``.

    """
    alpha_param = iap.handle_continuous_param(
        alpha, "alpha",
        value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
    strength_param = iap.handle_continuous_param(
        strength, "strength",
        value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        strength_sample = strength_param.draw_sample(
            random_state=random_state_func)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [-1-strength_sample, 0-strength_sample, 0],
            [0-strength_sample, 1, 0+strength_sample],
            [0, 0+strength_sample, 1+strength_sample]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Convolve(
        create_matrices,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


# TODO add tests
# TODO move this to edges.py?
def EdgeDetect(alpha=0, name=None, deterministic=False, random_state=None):
    """
    Generate a black & white edge image and alpha-blend it with the input image.

    dtype support::

        See ``imgaug.augmenters.convolutional.Convolve``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the edge image. At ``0.0``, only the original
        image is visible, at ``1.0`` only the edge image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.EdgeDetect(alpha=(0.0, 1.0))

    Detect edges in an image, mark them as black (non-edge) and white (edges)
    and alpha-blend the result with the original input image using a random
    blending factor between ``0%`` and ``100%``.

    """
    alpha_param = iap.handle_continuous_param(
        alpha, "alpha",
        value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(_image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        matrix_effect = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )
        return [matrix] * nb_channels

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Convolve(
        create_matrices,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


# TODO add tests
# TODO merge EdgeDetect and DirectedEdgeDetect?
# TODO deprecate and rename to AngledEdgeDetect
# TODO rename arg "direction" to "angle"
# TODO change direction/angle value range to (0, 360)
# TODO move this to edges.py?
def DirectedEdgeDetect(alpha=0, direction=(0.0, 1.0),
                       name=None, deterministic=False, random_state=None):
    """
    Detect edges from specified angles and alpha-blend with the input image.

    This augmenter first detects edges along a certain angle.
    Usually, edges are detected in x- or y-direction, while here the edge
    detection kernel is rotated to match a specified angle.
    The result of applying the kernel is a black (non-edges) and white (edges)
    image. That image is alpha-blended with the input image.

    dtype support::

        See ``imgaug.augmenters.convolutional.Convolve``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Blending factor of the edge image. At ``0.0``, only the original
        image is visible, at ``1.0`` only the edge image is visible.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` per image.
            * If a list, a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from that
              parameter per image.

    direction : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Angle (in degrees) of edges to pronounce, where ``0`` represents
        ``0`` degrees and ``1.0`` represents 360 degrees (both clockwise,
        starting at the top). Default value is ``(0.0, 1.0)``, i.e. pick a
        random angle per image.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value will be sampled from the
              interval ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=0)

    Turn input images into edge images in which edges are detected from
    the top side of the image (i.e. the top sides of horizontal edges are
    part of the edge image, while vertical edges are ignored).

    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=90/360)

    Same as before, but edges are detected from the right. Horizontal edges
    are now ignored.

    >>> aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=(0.0, 1.0))

    Same as before, but edges are detected from a random angle sampled
    uniformly from the interval ``[0deg, 360deg]``.

    >>> aug = iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=0)

    Similar to the previous examples, but here the edge image is alpha-blended
    with the input image. The result is a mixture between the edge image and
    the input image. The blending factor is randomly sampled between ``0%``
    and ``30%``.

    """
    alpha_param = iap.handle_continuous_param(
        alpha, "alpha",
        value_range=(0, 1.0), tuple_to_uniform=True, list_to_choice=True)
    direction_param = iap.handle_continuous_param(
        direction, "direction",
        value_range=None, tuple_to_uniform=True, list_to_choice=True)

    def create_matrices(_image, nb_channels, random_state_func):
        alpha_sample = alpha_param.draw_sample(random_state=random_state_func)
        ia.do_assert(0 <= alpha_sample <= 1.0)
        direction_sample = direction_param.draw_sample(
            random_state=random_state_func)

        deg = int(direction_sample * 360) % 360
        rad = np.deg2rad(deg)
        x = np.cos(rad - 0.5*np.pi)
        y = np.sin(rad - 0.5*np.pi)
        direction_vector = np.array([x, y])

        matrix_effect = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        for x, y in itertools.product([-1, 0, 1], [-1, 0, 1]):
            if (x, y) != (0, 0):
                cell_vector = np.array([x, y])
                distance_deg = np.rad2deg(
                    ia.angle_between_vectors(cell_vector, direction_vector))
                distance = distance_deg / 180
                similarity = (1 - distance)**4
                matrix_effect[y+1, x+1] = similarity
        matrix_effect = matrix_effect / np.sum(matrix_effect)
        matrix_effect = matrix_effect * (-1)
        matrix_effect[1, 1] = 1

        matrix_nochange = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

        matrix = (
            (1-alpha_sample) * matrix_nochange
            + alpha_sample * matrix_effect
        )

        return [matrix] * nb_channels

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return Convolve(
        create_matrices,
        name=name,
        deterministic=deterministic,
        random_state=random_state)
