"""
Augmenters that perform simple arithmetic changes.

Do not import directly from this file, as the categorization is not final.
Use instead::

    from imgaug import augmenters as iaa

and then e.g.::

    seq = iaa.Sequential([iaa.Add((-5, 5)), iaa.Multiply((0.9, 1.1))])

List of augmenters:

    * Add
    * AddElementwise
    * AdditiveGaussianNoise
    * AdditiveLaplaceNoise
    * AdditivePoissonNoise
    * Multiply
    * MultiplyElementwise
    * Dropout
    * CoarseDropout
    * ReplaceElementwise
    * ImpulseNoise
    * SaltAndPepper
    * CoarseSaltAndPepper
    * Salt
    * CoarseSalt
    * Pepper
    * CoarsePepper
    * Invert
    * ContrastNormalization
    * JpegCompression

"""
from __future__ import print_function, division, absolute_import

from PIL import Image as PIL_Image
import imageio
import tempfile
import numpy as np
import cv2

from . import meta
import imgaug as ia
from .. import parameters as iap
from .. import dtypes as iadt


class Add(meta.Augmenter):
    """
    Add a value to all pixels in an image.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: yes; tested

    Parameters
    ----------
    value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Value to add to all pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a value will be sampled per
              image from that parameter.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Add(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.Add((-10, 10))

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    Adds a value from the discrete interval ``[-10..10]`` to all pixels of
    input images. The exact value is sampled per image *and* channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Add, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.value = iap.handle_continuous_param(
            value, "value", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float64", "float96", "float128",
                                     "float256"],
                         augmenter=self)

        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        nb_images = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)
        rss = ia.derive_random_states(random_state, 2)
        value_samples = self.value.draw_samples(
            (nb_images, nb_channels_max), random_state=rss[0])
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[1])

        gen = enumerate(zip(images, value_samples, per_channel_samples,
                            input_dtypes))
        for i, (image, value_samples_i, per_channel_samples_i, input_dtype) in gen:
            nb_channels = image.shape[2]

            # Example code to directly add images via image+sample (uint8 only)
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.int16)
            #     value_samples_i = value_samples_i.astype(np.int16)
            #     for c, value in enumerate(value_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] + value, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.int16)
            #         + value_samples_i[0].astype(np.int16),
            #         0, 255
            #     ).astype(np.uint8)

            if image.dtype.name == "uint8":
                # Using this LUT approach is significantly faster than the
                # else-block code (around 3-4x speedup) and is still faster
                # than the simpler image+sample approach without LUT (about
                # 10% at 64x64 and about 2x at 224x224 -- maybe dependent on
                # installed BLAS libraries?)
                value_samples_i = np.clip(
                    np.round(value_samples_i), -255, 255).astype(np.int16)
                value_range = np.arange(0, 256, dtype=np.int16)
                if per_channel_samples_i > 0.5:
                    result = []
                    tables = np.tile(
                        value_range[np.newaxis, :],
                        (nb_channels, 1)
                    ) + value_samples_i[0:nb_channels, np.newaxis]
                    tables = np.clip(tables, 0, 255).astype(image.dtype)
                    for c, table in enumerate(tables):
                        arr_aug = cv2.LUT(image[..., c], table)
                        result.append(arr_aug[..., np.newaxis])
                    images[i] = np.concatenate(result, axis=2)
                else:
                    table = value_range + value_samples_i[0]
                    image_aug = cv2.LUT(
                        image,
                        np.clip(table, 0, 255).astype(image.dtype))
                    if image_aug.ndim == 2:
                        image_aug = image_aug[..., np.newaxis]
                    images[i] = image_aug
            else:
                if per_channel_samples_i > 0.5:
                    value = value_samples_i[0:nb_channels].reshape(
                        (1, 1, nb_channels))
                else:
                    value = value_samples_i[0:1].reshape((1, 1, 1))

                # We limit here the value range of the value parameter to the
                # bytes in the image's dtype. This prevents overflow problems
                # and makes it less likely that the image has to be up-casted,
                # which again improves performance and saves memory. Note that
                # this also enables more dtypes for image inputs.
                # The downside is that the mul parameter is limited in its
                # value range.
                #
                # We need 2* the itemsize of the image here to allow to shift
                # the image's max value to the lowest possible value, e.g. for
                # uint8 it must allow for -255 to 255.
                itemsize = image.dtype.itemsize * 2
                dtype_target = np.dtype("%s%d" % (value.dtype.kind, itemsize))
                value = iadt.clip_to_dtype_value_range_(
                    value, dtype_target, validate=True)

                image, value = iadt.promote_array_dtypes_(
                    [image, value],
                    dtypes=[image.dtype, dtype_target],
                    increase_itemsize_factor=2)
                image = np.add(image, value, out=image, casting="no")

                image = iadt.restore_dtypes_(image, input_dtype)
                images[i] = image

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.value, self.per_channel]


# TODO merge this with Add
class AddElementwise(meta.Augmenter):
    """
    Add to the pixels of images values that are pixelwise randomly sampled.

    While the ``Add`` Augmenter samples one value to add *per image* (and
    optionally per channel), this augmenter samples different values per image
    and *per pixel* (and optionally per channel), i.e. intensities of
    neighbouring pixels may be increased/decreased by different amounts.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: yes; tested

    Parameters
    ----------
    value : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Value to add to the pixels.

            * If an int, exactly that value will always be used.
            * If a tuple ``(a, b)``, then values from the discrete interval
              ``[a..b]`` will be sampled per image and pixel.
            * If a list of integers, a random value will be sampled from the
              list per image and pixel.
            * If a ``StochasticParameter``, then values will be sampled per
              image and pixel from that parameter.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AddElementwise(10)

    Always adds a value of 10 to all channels of all pixels of all input
    images.

    >>> aug = iaa.AddElementwise((-10, 10))

    Samples per image and pixel a value from the discrete interval
    ``[-10..10]`` and adds that value to the respective pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=True)

    Samples per image, pixel *and also channel* a value from the discrete
    interval ``[-10..10]`` and adds it to the respective pixel's channel value.
    Therefore, added values may differ between channels of the same pixel.

    >>> aug = iaa.AddElementwise((-10, 10), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(AddElementwise, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        # TODO open to continous, similar to Add
        self.value = iap.handle_discrete_param(
            value, "value", value_range=(-255, 255), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float64", "float96", "float128",
                                     "float256"],
                         augmenter=self)

        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, nb_images+1)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[-1])

        gen = enumerate(zip(images, per_channel_samples, rss[:-1],
                            input_dtypes))
        for i, (image, per_channel_samples_i, rs, input_dtype) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            value = self.value.draw_samples(sample_shape, random_state=rs)

            if image.dtype.name == "uint8":
                # This special uint8 block is around 60-100% faster than the
                # else-block further below (more speedup for smaller images).
                #
                # Also tested to instead compute min/max of image and value
                # and then only convert image/value dtype if actually
                # necessary, but that was like 20-30% slower, even for 224x224
                # images.
                #
                if value.dtype.kind == "f":
                    value = np.round(value)

                image = image.astype(np.int16)
                value = np.clip(value, -255, 255).astype(np.int16)

                image_aug = image + value
                image_aug = np.clip(image_aug, 0, 255).astype(np.uint8)

                images[i] = image_aug
            else:
                # We limit here the value range of the value parameter to the
                # bytes in the image's dtype. This prevents overflow problems
                # and makes it less likely that the image has to be up-casted,
                # which again improves performance and saves memory. Note that
                # this also enables more dtypes for image inputs.
                # The downside is that the mul parameter is limited in its
                # value range.
                #
                # We need 2* the itemsize of the image here to allow to shift
                # the image's max value to the lowest possible value, e.g. for
                # uint8 it must allow for -255 to 255.
                itemsize = image.dtype.itemsize * 2
                dtype_target = np.dtype("%s%d" % (value.dtype.kind, itemsize))
                value = iadt.clip_to_dtype_value_range_(value, dtype_target,
                                                        validate=100)

                if value.shape[2] == 1:
                    value = np.tile(value, (1, 1, nb_channels))

                image, value = iadt.promote_array_dtypes_(
                    [image, value],
                    dtypes=[image.dtype, dtype_target],
                    increase_itemsize_factor=2)
                image = np.add(image, value, out=image, casting="no")
                image = iadt.restore_dtypes_(image, input_dtype)
                images[i] = image

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.value, self.per_channel]


# TODO rename to AddGaussianNoise?
# TODO examples say that iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)) samples
#      the scale from the uniform dist. per image, but is that still the case?
#      AddElementwise seems to now sample once for all images, which should
#      lead to a single scale value.
def AdditiveGaussianNoise(loc=0, scale=0, per_channel=False,
                          name=None, deterministic=False, random_state=None):
    """
    Add noise sampled from gaussian distributions elementwise to images.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    dtype support::

        See ``imgaug.augmenters.arithmetic.AddElementwise``.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the normal distribution from which the noise is sampled.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the normal distribution that generates the noise.
        Must be ``>=0``. If ``0`` then `loc` will simply be added to all
        pixels.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))

    Adds gaussian noise from the distribution ``N(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=True)

    Adds gaussian noise from the distribution ``N(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    loc2 = iap.handle_continuous_param(
        loc, "loc", value_range=None, tuple_to_uniform=True,
        list_to_choice=True)
    scale2 = iap.handle_continuous_param(
        scale, "scale", value_range=(0, None), tuple_to_uniform=True,
        list_to_choice=True)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AddElementwise(
        iap.Normal(loc=loc2, scale=scale2),
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


# TODO rename to AddLaplaceNoise?
def AdditiveLaplaceNoise(loc=0, scale=0, per_channel=False,
                         name=None, deterministic=False, random_state=None):
    """
    Add noise sampled from laplace distributions elementwise to images.

    The laplace distribution is similar to the gaussian distribution, but
    puts more weight on the long tail. Hence, this noise will add more
    outliers (very high/low values). It is somewhere between gaussian noise and
    salt and pepper noise.

    Values of around ``255 * 0.05`` for `scale` lead to visible noise (for
    ``uint8``).
    Values of around ``255 * 0.10`` for `scale` lead to very visible
    noise (for ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    dtype support::

        See ``imgaug.augmenters.arithmetic.AddElementwise``.

    Parameters
    ----------
    loc : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean of the laplace distribution that generates the noise.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    scale : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Standard deviation of the laplace distribution that generates the noise.
        Must be ``>=0``. If ``0`` then only `loc` will be used.
        Recommended to be around ``255*0.05``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255))

    Adds laplace noise from the distribution ``Laplace(0, s)`` to images,
    where ``s`` is sampled per image from the interval ``[0, 0.1*255]``.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=True)

    Adds laplace noise from the distribution ``Laplace(0, 0.1*255)`` to images,
    where the noise value is different per image and pixel *and* channel (e.g.
    a different one for the red, green and blue channels of the same pixel).
    This leads to "colorful" noise.

    >>> aug = iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    loc2 = iap.handle_continuous_param(
        loc, "loc", value_range=None, tuple_to_uniform=True,
        list_to_choice=True)
    scale2 = iap.handle_continuous_param(
        scale, "scale", value_range=(0, None), tuple_to_uniform=True,
        list_to_choice=True)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AddElementwise(
        iap.Laplace(loc=loc2, scale=scale2),
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


# TODO rename to AddPoissonNoise?
def AdditivePoissonNoise(lam=0, per_channel=False,
                         name=None, deterministic=False, random_state=None):
    """
    Add noise sampled from poisson distributions elementwise to images.

    Poisson noise is comparable to gaussian noise, as e.g. generated via
    ``AdditiveGaussianNoise``. As poisson distributions produce only positive
    numbers, the sign of the sampled values are here randomly flipped.

    Values of around ``10.0`` for `lam` lead to visible noise (for ``uint8``).
    Values of around ``20.0`` for `lam` lead to very visible noise (for
    ``uint8``).
    It is recommended to usually set `per_channel` to ``True``.

    This augmenter samples and adds noise elementwise, i.e. it can add
    different noise values to neighbouring pixels and is comparable
    to ``AddElementwise``.

    dtype support::

        See ``imgaug.augmenters.arithmetic.AddElementwise``.

    Parameters
    ----------
    lam : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Lambda parameter of the poisson distribution. Must be ``>=0``.
        Recommended values are around ``0.0`` to ``10.0``.

            * If a number, exactly that value will always be used.
            * If a tuple ``(a, b)``, a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, a value will be sampled from the
              parameter per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.AdditivePoissonNoise(lam=5.0)

    Adds poisson noise sampled from a poisson distribution with a ``lambda``
    parameter of ``5.0`` to images.
    The samples are drawn per image and pixel.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 10.0))

    Adds poisson noise sampled from ``Poisson(x)`` to images, where ``x`` is
    randomly sampled per image from the interval ``[0.0, 10.0]``.

    >>> aug = iaa.AdditivePoissonNoise(lam=5.0, per_channel=True)

    Adds poisson noise sampled from ``Poisson(5.0)`` to images,
    where the values are different per image and pixel *and* channel (e.g. a
    different one for red, green and blue channels for the same pixel).

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 10.0), per_channel=True)

    Adds poisson noise sampled from ``Poisson(x)`` to images,
    with ``x`` being sampled from ``uniform(0.0, 10.0)`` per image and
    channel. This is the *recommended* configuration.

    >>> aug = iaa.AdditivePoissonNoise(lam=(0.0, 10.0), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    lam2 = iap.handle_continuous_param(
        lam, "lam",
        value_range=(0, None), tuple_to_uniform=True, list_to_choice=True)

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return AddElementwise(
        iap.RandomSign(iap.Poisson(lam=lam2)),
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


class Multiply(meta.Augmenter):
    """
    Multiply all pixels in an image with a random value sampled once per image.

    This augmenter can be used to make images lighter or darker.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: yes; tested

        Note: tests were only conducted for rather small multipliers, around
        -10.0 to +10.0.

        In general, the multipliers sampled from `mul` must be in a value
        range that corresponds to the input image's dtype. E.g. if the input
        image has dtype uint16 and the samples generated from `mul` are
        ``float64``, this augmenter will still force all samples to be within
        the value range of ``float16``, as it has the same number of
        bytes (two) as ``uint16``. This is done to make overflows less likely
        to occur.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value with which to multiply the pixel values in each image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and used for all pixels.
            * If a list, then a random value will be sampled from that list per
              image.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Multiply(2.0)

    Multiplies all images by a factor of ``2``, making the images significantly
    brighter.

    >>> aug = iaa.Multiply((0.5, 1.5))

    Multiplies images by a random value sampled uniformly from the interval
    ``[0.5, 1.5]``, making some images darker and others brighter.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=True)

    Identical to the previous example, but the sampled multipliers differ by
    image *and* channel, instead of only by image.

    >>> aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, mul=1.0, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(Multiply, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.mul = iap.handle_continuous_param(
            mul, "mul", value_range=None, tuple_to_uniform=True,
            list_to_choice=True)
        self.per_channel = iap.handle_probability_param(
            per_channel, "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float64", "float96", "float128",
                                     "float256"],
                         augmenter=self)

        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        nb_images = len(images)
        nb_channels_max = meta.estimate_max_number_of_channels(images)
        rss = ia.derive_random_states(random_state, 2)
        mul_samples = self.mul.draw_samples(
            (nb_images, nb_channels_max), random_state=rss[0])
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[1])

        gen = enumerate(zip(images, mul_samples, per_channel_samples,
                            input_dtypes))
        for i, (image, mul_samples_i, per_channel_samples_i, input_dtype) in gen:
            nb_channels = image.shape[2]

            # Example code to directly multiply images via image*sample
            # (uint8 only) -- apparently slower than LUT
            # if per_channel_samples_i > 0.5:
            #     result = []
            #     image = image.astype(np.float32)
            #     mul_samples_i = mul_samples_i.astype(np.float32)
            #     for c, mul in enumerate(mul_samples_i[0:nb_channels]):
            #         result.append(
            #             np.clip(
            #                 image[..., c:c+1] * mul, 0, 255
            #             ).astype(np.uint8))
            #     images[i] = np.concatenate(result, axis=2)
            # else:
            #     images[i] = np.clip(
            #         image.astype(np.float32)
            #         * mul_samples_i[0].astype(np.float32),
            #         0, 255
            #     ).astype(np.uint8)

            if image.dtype.name == "uint8":
                # Using this LUT approach is significantly faster than
                # else-block code (more than 10x speedup) and is still faster
                # than the simpler image*sample approach without LUT (1.5-3x
                # speedup, maybe dependent on installed BLAS libraries?)
                value_range = np.arange(0, 256, dtype=np.float32)
                if per_channel_samples_i > 0.5:
                    result = []
                    mul_samples_i = mul_samples_i.astype(np.float32)
                    tables = np.tile(
                        value_range[np.newaxis, :], (nb_channels, 1)) \
                        * mul_samples_i[0:nb_channels, np.newaxis]
                    tables = np.clip(tables, 0, 255).astype(image.dtype)
                    for c, table in enumerate(tables):
                        arr_aug = cv2.LUT(image[..., c], table)
                        result.append(arr_aug[..., np.newaxis])
                    images[i] = np.concatenate(result, axis=2)
                else:
                    table = (value_range
                             * mul_samples_i[0].astype(np.float32))
                    image_aug = cv2.LUT(
                        image, np.clip(table, 0, 255).astype(image.dtype))
                    if image_aug.ndim == 2:
                        image_aug = image_aug[..., np.newaxis]
                    images[i] = image_aug
            else:
                # TODO estimate via image min/max values whether a resolution
                #      increase is necessary

                if per_channel_samples_i > 0.5:
                    mul = mul_samples_i[0:nb_channels].reshape(
                        (1, 1, nb_channels))
                else:
                    mul = mul_samples_i[0:1].reshape((1, 1, 1))

                mul_min = np.min(mul)
                mul_max = np.max(mul)
                is_not_increasing_value_range = (
                        (-1 <= mul_min <= 1)
                        and (-1 <= mul_max <= 1))

                # We limit here the value range of the mul parameter to the
                # bytes in the image's dtype. This prevents overflow problems
                # and makes it less likely that the image has to be up-casted,
                # which again improves performance and saves memory. Note that
                # this also enables more dtypes for image inputs.
                # The downside is that the mul parameter is limited in its
                # value range.
                itemsize = max(
                    image.dtype.itemsize,
                    2 if mul.dtype.kind == "f" else 1
                )  # float min itemsize is 2 not 1
                dtype_target = np.dtype("%s%d" % (mul.dtype.kind, itemsize))
                mul = iadt.clip_to_dtype_value_range_(mul, dtype_target,
                                                      validate=True)

                image, mul = iadt.promote_array_dtypes_(
                    [image, mul],
                    dtypes=[image.dtype, dtype_target],
                    increase_itemsize_factor=(
                        1 if is_not_increasing_value_range else 2)
                )
                image = np.multiply(image, mul, out=image, casting="no")

                image = iadt.restore_dtypes_(image, input_dtype)
                images[i] = image

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul, self.per_channel]


# TODO merge with Multiply
class MultiplyElementwise(meta.Augmenter):
    """
    Multiply image pixels with values that are pixelwise randomly sampled.

    While the ``Multiply`` Augmenter uses a constant multiplier *per
    image* (and optionally channel), this augmenter samples the multipliers
    to use per image and *per pixel* (and optionally per channel).

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: no
        * ``int64``: no
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: no
        * ``float128``: no
        * ``bool``: yes; tested

        Note: tests were only conducted for rather small multipliers, around
        -10.0 to +10.0.

        In general, the multipliers sampled from `mul` must be in a value
        range that corresponds to the input image's dtype. E.g. if the input
        image has dtype uint16 and the samples generated from `mul` are
        ``float64``, this augmenter will still force all samples to be within
        the value range of ``float16``, as it has the same number of
        bytes (two) as ``uint16``. This is done to make overflows less likely
        to occur.

    Parameters
    ----------
    mul : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        The value with which to multiply pixel values in the image.

            * If a number, then that value will always be used.
            * If a tuple ``(a, b)``, then a value from the interval ``[a, b]``
              will be sampled per image and pixel.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then that parameter will be used to
              sample a new value per image and pixel.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.MultiplyElementwise(2.0)

    Multiply all images by a factor of ``2.0``, making them significantly
    bighter.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5))

    Samples per image and pixel uniformly a value from the interval
    ``[0.5, 1.5]`` and multiplies the pixel with that value.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)

    Samples per image and pixel *and channel* uniformly a value from the
    interval ``[0.5, 1.5]`` and multiplies the pixel with that value. Therefore,
    used multipliers may differ between channels of the same pixel.

    >>> aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """

    def __init__(self, mul=1.0, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(MultiplyElementwise, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.mul = iap.handle_continuous_param(
            mul, "mul",
            value_range=None, tuple_to_uniform=True, list_to_choice=True)
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16",
                                  "int8", "int16",
                                  "float16", "float32"],
                         disallowed=["uint32", "uint64", "uint128", "uint256",
                                     "int32", "int64", "int128", "int256",
                                     "float64", "float96", "float128", "float256"],
                         augmenter=self)

        input_dtypes = iadt.copy_dtypes_for_restore(images, force_list=True)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, nb_images+1)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[-1])
        is_mul_binomial = isinstance(self.mul, iap.Binomial) or (
            isinstance(self.mul, iap.FromLowerResolution)
            and isinstance(self.mul.other_param, iap.Binomial)
        )

        gen = enumerate(zip(images, per_channel_samples, rss[:-1],
                            input_dtypes))
        for i, (image, per_channel_samples_i, rs, input_dtype) in gen:
            height, width, nb_channels = image.shape
            sample_shape = (height,
                            width,
                            nb_channels if per_channel_samples_i > 0.5 else 1)
            mul = self.mul.draw_samples(sample_shape, random_state=rs)
            # TODO let Binomial return boolean mask directly instead of [0, 1]
            #      integers?

            # hack to improve performance for Dropout and CoarseDropout
            # converts mul samples to mask if mul is binomial
            if mul.dtype.kind != "b" and is_mul_binomial:
                mul = mul.astype(bool, copy=False)

            if mul.dtype.kind == "b":
                images[i] *= mul
            elif image.dtype.name == "uint8":
                # This special uint8 block is around 60-100% faster than the
                # else-block further below (more speedup for larger images).
                #
                if mul.dtype.kind == "f":
                    # interestingly, float32 is here significantly faster than
                    # float16
                    # TODO is that system dependent?
                    # TODO does that affect int8-int32 too?
                    mul = mul.astype(np.float32, copy=False)
                    image_aug = image.astype(np.float32)
                else:
                    mul = mul.astype(np.int16, copy=False)
                    image_aug = image.astype(np.int16)

                image_aug = np.multiply(image_aug, mul, casting="no",
                                        out=image_aug)
                images[i] = iadt.restore_dtypes_(image_aug, np.uint8,
                                                 round=False)
            else:
                # TODO maybe introduce to stochastic parameters some way to
                #      get the possible min/max values, could make things
                #      faster for dropout to get 0/1 min/max from the binomial
                mul_min = np.min(mul)
                mul_max = np.max(mul)
                is_not_increasing_value_range = (
                    (-1 <= mul_min <= 1) and (-1 <= mul_max <= 1))

                # We limit here the value range of the mul parameter to the
                # bytes in the image's dtype. This prevents overflow problems
                # and makes it less likely that the image has to be up-casted,
                # which again improves performance and saves memory. Note that
                # this also enables more dtypes for image inputs.
                # The downside is that the mul parameter is limited in its
                # value range.
                itemsize = max(
                    image.dtype.itemsize,
                    2 if mul.dtype.kind == "f" else 1
                )  # float min itemsize is 2
                dtype_target = np.dtype("%s%d" % (mul.dtype.kind, itemsize))
                mul = iadt.clip_to_dtype_value_range_(
                    mul, dtype_target,
                    validate=True, validate_values=(mul_min, mul_max))

                if mul.shape[2] == 1:
                    mul = np.tile(mul, (1, 1, nb_channels))

                image, mul = iadt.promote_array_dtypes_(
                    [image, mul],
                    dtypes=[image, dtype_target],
                    increase_itemsize_factor=(
                        1 if is_not_increasing_value_range else 2)
                )
                image = np.multiply(image, mul, out=image, casting="no")
                image = iadt.restore_dtypes_(image, input_dtype)
                images[i] = image

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.mul, self.per_channel]


# TODO verify that (a, b) still leads to a p being sampled per image and not
#      per batch
def Dropout(p=0, per_channel=False,
            name=None, deterministic=False, random_state=None):
    """
    Set a fraction of pixels in images to zero.

    dtype support::

        See ``imgaug.augmenters.arithmetic.MultiplyElementwise``.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. to set it to zero).

            * If a float, then that value will be used for all images. A value
              of ``1.0`` would mean that all pixels will be dropped
              and ``0.0`` that no pixels will be dropped. A value of ``0.05``
              corresponds to ``5`` percent of all pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the pixel's
              dropout probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Dropout(0.02)

    Drops ``2`` percent of all pixels.

    >>> aug = iaa.Dropout((0.0, 0.05))

    Drops in each image a random fraction of all pixels, where the fraction
    is uniformly sampled from the interval ``[0.0, 0.05]``.

    >>> aug = iaa.Dropout(0.02, per_channel=True)

    Drops ``2`` percent of all pixels in a channelwise fashion, i.e. it is
    unlikely for any pixel to have all channels set to zero (black pixels).

    >>> aug = iaa.Dropout(0.02, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for ``50`` percent of all images.

    """
    # TODO add list as an option
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception(
            "Expected p to be float or int or StochasticParameter, got %s." % (
                type(p),))

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(
        p2,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


# TODO add similar cutout augmenter
# TODO invert size_p and size_percent so that larger values denote larger
#      areas being dropped instead of the opposite way around
def CoarseDropout(p=0, size_px=None, size_percent=None, per_channel=False,
                  min_size=4,
                  name=None, deterministic=False, random_state=None):
    """
    Set rectangular areas within images to zero.

    In contrast to ``Dropout``, these areas can have larger sizes.
    (E.g. you might end up with three large black rectangles in an image.)
    Note that the current implementation leads to correlated sizes,
    so if e.g. there is any thin and high rectangle that is dropped, there is
    a high likelihood that all other dropped areas are also thin and high.

    This method is implemented by generating the dropout mask at a
    lower resolution (than the image has) and then upsampling the mask
    before dropping the pixels.

    This augmenter is similar to Cutout. Usually, cutout is defined as an
    operation that drops exactly one rectangle from an image, while here
    ``CoarseDropout`` can drop multiple rectangles (with some correlation
    between the sizes of these rectangles).

    dtype support::

        See ``imgaug.augmenters.arithmetic.MultiplyElementwise``.

    Parameters
    ----------
    p : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The probability of any pixel being dropped (i.e. set to zero) in
        the lower-resolution dropout mask.

            * If a float, then that value will be used for all pixels. A value
              of ``1.0`` would mean, that all pixels will be dropped. A value
              of ``0.0`` would lead to no pixels being dropped.
            * If a tuple ``(a, b)``, then a value ``p`` will be sampled from
              the interval ``[a, b]`` per image and be used as the dropout
              probability.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per pixel whether it should be *kept* (sampled value
              of ``>0.5``) or shouldn't be kept (sampled value of ``<=0.5``).
              If you instead want to provide the probability as a stochastic
              parameter, you can usually do ``imgaug.parameters.Binomial(1-p)``
              to convert parameter `p` to a 0/1 representation.

    size_px : None or int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The dropout mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : None or float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the dropout
        mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being dropped (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being dropped.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5)

    Drops ``2`` percent of all pixels on a lower-resolution image that has
    ``50`` percent of the original image's size, leading to dropped areas that
    have roughly ``2x2`` pixels size.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.05, 0.5))

    Generates a dropout mask at ``5`` to ``50`` percent of each input image's
    size. In that mask, ``0`` to ``5`` percent of all pixels are marked as
    being dropped. The mask is afterwards projected to the input image's
    size to apply the actual dropout operation.

    >>> aug = iaa.CoarseDropout((0.0, 0.05), size_px=(2, 16))

    Same as the previous example, but the lower resolution image has ``2`` to
    ``16`` pixels size. On images of e.g. ``224x224` pixels in size this would
    lead to fairly large areas being dropped (height/width of ``224/2`` to
    ``224/16``).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=True)

    Drops ``2`` percent of all pixels at ``50`` percent resolution (``2x2``
    sizes) in a channel-wise fashion, i.e. it is unlikely for any pixel to
    have all channels set to zero (black pixels).

    >>> aug = iaa.CoarseDropout(0.02, size_percent=0.5, per_channel=0.5)

    Same as the previous example, but the `per_channel` feature is only active
    for ``50`` percent of all images.

    """
    if ia.is_single_number(p):
        p2 = iap.Binomial(1 - p)
    elif ia.is_iterable(p):
        ia.do_assert(len(p) == 2)
        ia.do_assert(p[0] < p[1])
        ia.do_assert(0 <= p[0] <= 1.0)
        ia.do_assert(0 <= p[1] <= 1.0)
        p2 = iap.Binomial(iap.Uniform(1 - p[1], 1 - p[0]))
    elif isinstance(p, iap.StochasticParameter):
        p2 = p
    else:
        raise Exception("Expected p to be float or int or StochasticParameter, "
                        "got %s." % (type(p),))

    if size_px is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_px=size_px,
                                     min_size=min_size)
    elif size_percent is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_percent=size_percent,
                                     min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(
        p3,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


class ReplaceElementwise(meta.Augmenter):
    """
    Replace pixels in an image with new values.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: no (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: no
        * ``bool``: yes; tested

        - (1) ``uint64`` is currently not supported, because
              :func:`imgaug.dtypes.clip_to_dtype_value_range_()` does not
              support it, which again is because numpy.clip() seems to not
              support it.

    Parameters
    ----------
    mask : float or tuple of float or list of float or imgaug.parameters.StochasticParameter
        Mask that indicates the pixels that are supposed to be replaced.
        The mask will be binarized using a threshold of ``0.5``. A value
        of ``1`` then indicates a pixel that is supposed to be replaced.

            * If this is a float, then that value will be used as the
              probability of being a ``1`` in the mask (sampled per image and
              pixel) and hence being replaced.
            * If a tuple ``(a, b)``, then the probability will be uniformly
              sampled per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample a mask per image.

    replacement : number or tuple of number or list of number or imgaug.parameters.StochasticParameter
        The replacement to use at all locations that are marked as ``1`` in
        the mask.

            * If this is a number, then that value will always be used as the
              replacement.
            * If a tuple ``(a, b)``, then the replacement will be sampled
              uniformly per image and pixel from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and pixel.
            * If a ``StochasticParameter``, then this parameter will be used
              sample replacement values per image and pixel.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = ReplaceElementwise(0.05, [0, 255])

    Replaces ``5`` percent of all pixels in each image by either ``0``
    or ``255``.

    """

    def __init__(self, mask, replacement, per_channel=False,
                 name=None, deterministic=False, random_state=None):
        super(ReplaceElementwise, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.mask = iap.handle_probability_param(
            mask, "mask", tuple_to_uniform=True, list_to_choice=True)
        self.replacement = iap.handle_continuous_param(replacement,
                                                       "replacement")
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16", "uint32",
                                  "int8", "int16", "int32", "int64",
                                  "float16", "float32", "float64"],
                         disallowed=["uint64", "uint128", "uint256",
                                     "int128", "int256",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        rss = ia.derive_random_states(random_state, 2*nb_images+1)
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[-1])

        gen = zip(images, per_channel_samples, rss[:-1:2], rss[1:-1:2])
        for image, per_channel_i, rs_mask, rs_replacement in gen:
            height, width, nb_channels = image.shape
            sampling_shape = (height,
                              width,
                              nb_channels if per_channel_i > 0.5 else 1)
            mask_samples = self.mask.draw_samples(sampling_shape,
                                                  random_state=rs_mask)

            # This is slightly faster (~20%) for masks that are True at many
            # locations, but slower (~50%) for masks with few Trues, which is
            # probably the more common use-case:
            #
            # replacement_samples = self.replacement.draw_samples(
            #     sampling_shape, random_state=rs_replacement)
            #
            # # round, this makes 0.2 e.g. become 0 in case of boolean
            # # image (otherwise replacing values with 0.2 would
            # # lead to True instead of False).
            # if (image.dtype.kind in ["i", "u", "b"]
            #         and replacement_samples.dtype.kind == "f"):
            #     replacement_samples = np.round(replacement_samples)
            #
            # replacement_samples = iadt.clip_to_dtype_value_range_(
            #     replacement_samples, image.dtype, validate=False)
            # replacement_samples = replacement_samples.astype(
            #     image.dtype, copy=False)
            #
            # if sampling_shape[2] == 1:
            #     mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
            #     replacement_samples = np.tile(
            #         replacement_samples, (1, 1, nb_channels))
            # mask_thresh = mask_samples > 0.5
            # image[mask_thresh] = replacement_samples[mask_thresh]

            if sampling_shape[2] == 1:
                mask_samples = np.tile(mask_samples, (1, 1, nb_channels))
            mask_thresh = mask_samples > 0.5

            # TODO add separate per_channels for mask and replacement
            # TODO add test that replacement with per_channel=False is not
            #      sampled per channel
            if per_channel_i <= 0.5:
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_thresh[:, :, 0])),),
                    random_state=rs_replacement)
                # important here to use repeat instead of tile. repeat
                # converts e.g. [0, 1, 2] to [0, 0, 1, 1, 2, 2], while tile
                # leads to [0, 1, 2, 0, 1, 2]. The assignment below iterates
                # over each channel and pixel simultaneously, *not* first
                # over all pixels of channel 0, then all pixels in
                # channel 1, ...
                replacement_samples = np.repeat(
                    replacement_samples, mask_thresh.shape[2])
            else:
                replacement_samples = self.replacement.draw_samples(
                    (int(np.sum(mask_thresh)),), random_state=rs_replacement)

            # round, this makes 0.2 e.g. become 0 in case of boolean
            # image (otherwise replacing values with 0.2 would lead to True
            # instead of False).
            if (image.dtype.kind in ["i", "u", "b"]
                    and replacement_samples.dtype.kind == "f"):
                replacement_samples = np.round(replacement_samples)

            replacement_samples = iadt.clip_to_dtype_value_range_(
                replacement_samples, image.dtype, validate=False)
            replacement_samples = replacement_samples.astype(
                image.dtype, copy=False)

            image[mask_thresh] = replacement_samples

        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.mask, self.replacement, self.per_channel]


def ImpulseNoise(p=0, name=None, deterministic=False, random_state=None):
    """
    Apply impulse noise to images.

    This is identical to ``SaltAndPepper``, except that `per_channel` is
    always set to ``True``.

    dtype support::

        See ``imgaug.augmenters.arithmetic.SaltAndPepper``.

    """
    return SaltAndPepper(
        p=p,
        per_channel=True,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


def SaltAndPepper(p=0, per_channel=False,
                  name=None, deterministic=False, random_state=None):
    """
    Replace pixels in images with salt/pepper noise (white/black-ish colors).

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt and pepper noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.SaltAndPepper(0.05)

    Replaces ``5`` percent of all pixels with salt and pepper noise (white-ish
    and black-ish colors).

    """
    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=p,
        replacement=iap.Beta(0.5, 0.5) * 255,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def CoarseSaltAndPepper(p=0, size_px=None, size_percent=None,
                        per_channel=False, min_size=4,
                        name=None, deterministic=False, random_state=None):
    """
    Replace rectangular areas in images with white/black-ish pixel noise.

    This adds salt and pepper noise (noisy white-ish and black-ish pixels) to
    rectangular areas within the image. Note that this means that within these
    rectangular areas the color varies instead of each rectangle having only
    one color.

    See also the similar ``CoarseDropout``.

    TODO replace dtype support with uint8 only, because replacement is
         geared towards that value range

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt/pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt and pepper noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))

    Marks ``5`` percent of all pixels in a mask to be replaced by salt/pepper
    noise. The mask has ``1`` to ``10`` percent the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt/pepper noise.

    """
    mask = iap.handle_probability_param(
        p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement = iap.Beta(0.5, 0.5) * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def Salt(p=0, per_channel=False,
         name=None, deterministic=False, random_state=None):
    """
    Replace pixels in images with salt noise, i.e. white-ish pixels.

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel with salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with salt noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Salt(0.05)

    Replaces ``5`` percent of all pixels with salt noise (white-ish colors).

    """

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=True,
        mode="invert"
    ) + 0.5
    # FIXME max replacement seems to essentially never exceed 254
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=p,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


def CoarseSalt(p=0, size_px=None, size_percent=None, per_channel=False,
               min_size=4,
               name=None, deterministic=False, random_state=None):
    """
    Replace rectangular areas in images with white-ish pixel noise.

    See also the similar ``CoarseSaltAndPepper``.

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to salt noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by salt noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum height and width of the low resolution mask. If
        `size_percent` or `size_px` leads to a lower value than this,
        `min_size` will be used instead. This should never have a value of
        less than ``2``, otherwise one may end up with a ``1x1`` low resolution
        mask, leading easily to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))

    Marks ``5`` percent of all pixels in a mask to be replaced by salt
    noise. The mask has ``1`` to ``10`` percent the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by salt noise.

    """
    mask = iap.handle_probability_param(
        p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=True,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state)


def Pepper(p=0, per_channel=False,
           name=None, deterministic=False, random_state=None):
    """
    Replace pixels in images with pepper noise, i.e. black-ish pixels.

    This is similar to dropout, but slower and the black pixels are not
    uniformly black.

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of replacing a pixel with pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a image-sized mask will be
              sampled from that parameter per image. Any value ``>0.5`` in
              that mask will be replaced with pepper noise.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Pepper(0.05)

    Replaces ``5`` percent of all pixels with pepper noise (black-ish colors).

    """

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=False,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=p,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


def CoarsePepper(p=0, size_px=None, size_percent=None, per_channel=False,
                 min_size=4,
                 name=None, deterministic=False, random_state=None):
    """
    Replace rectangular areas in images with black-ish pixel noise.

    dtype support::

        See ``imgaug.augmenters.arithmetic.ReplaceElementwise``.

    Parameters
    ----------
    p : float or tuple of float or list of float or imgaug.parameters.StochasticParameter, optional
        Probability of changing a pixel to pepper noise.

            * If a float, then that value will always be used as the
              probability.
            * If a tuple ``(a, b)``, then a probability will be sampled
              uniformly per image from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then a lower-resolution mask will
              be sampled from that parameter per image. Any value ``>0.5`` in
              that mask will denote a spatial location that is to be replaced
              by pepper noise.

    size_px : int or tuple of int or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask in absolute pixel dimensions.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_percent` must be set.
            * If an integer, then that size will always be used for both height
              and width. E.g. a value of ``3`` would lead to a ``3x3`` mask,
              which is then upsampled to ``HxW``, where ``H`` is the image size
              and ``W`` the image width.
            * If a tuple ``(a, b)``, then two values ``M``, ``N`` will be
              sampled from the discrete interval ``[a..b]``. The mask
              will then be generated at size ``MxN`` and upsampled to ``HxW``.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine the sizes. It is expected to be discrete.

    size_percent : float or tuple of float or imgaug.parameters.StochasticParameter, optional
        The size of the lower resolution image from which to sample the
        replacement mask *in percent* of the input image.
        Note that this means that *lower* values of this parameter lead to
        *larger* areas being replaced (as any pixel in the lower resolution
        image will correspond to a larger area at the original resolution).

            * If ``None`` then `size_px` must be set.
            * If a float, then that value will always be used as the percentage
              of the height and width (relative to the original size). E.g. for
              value ``p``, the mask will be sampled from ``(p*H)x(p*W)`` and
              later upsampled to ``HxW``.
            * If a tuple ``(a, b)``, then two values ``m``, ``n`` will be
              sampled from the interval ``(a, b)`` and used as the size
              fractions, i.e the mask size will be ``(m*H)x(n*W)``.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the percentage values. It is expected to be continuous.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_size : int, optional
        Minimum size of the low resolution mask, both width and height. If
        `size_percent` or `size_px` leads to a lower value than this, `min_size`
        will be used instead. This should never have a value of less than 2,
        otherwise one may end up with a ``1x1`` low resolution mask, leading
        easily to the whole image being replaced.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))

    Marks ``5`` percent of all pixels in a mask to be replaced by pepper
    noise. The mask has ``1`` to ``10`` percent the size of the input image.
    The mask is then upscaled to the input image size, leading to large
    rectangular areas being marked as to be replaced. These areas are then
    replaced in the input image by pepper noise.

    """
    mask = iap.handle_probability_param(
        p, "p", tuple_to_uniform=True, list_to_choice=True)

    if size_px is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        mask_low = iap.FromLowerResolution(
            other_param=mask, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    replacement01 = iap.ForceSign(
        iap.Beta(0.5, 0.5) - 0.5,
        positive=False,
        mode="invert"
    ) + 0.5
    replacement = replacement01 * 255

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return ReplaceElementwise(
        mask=mask_low,
        replacement=replacement,
        per_channel=per_channel,
        name=name,
        deterministic=deterministic,
        random_state=random_state
    )


class Invert(meta.Augmenter):
    """
    Invert all values in images, e.g. turn ``5`` into ``255-5=250``.

    For the standard value range of 0-255 it converts ``0`` to ``255``,
    ``255`` to ``0`` and ``10`` to ``(255-10)=245``.
    Let ``M`` be the maximum value possible, ``m`` the minimum value possible,
    ``v`` a value. Then the distance of ``v`` to ``m`` is ``d=abs(v-m)`` and
    the new value is given by ``v'=M-d``.

    dtype support::

        if (min_value=None and max_value=None)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: yes; tested
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: yes; tested
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: yes; tested
            * ``float128``: yes; tested
            * ``bool``: yes; tested

        if (min_value!=None or max_value!=None)::

            * ``uint8``: yes; fully tested
            * ``uint16``: yes; tested
            * ``uint32``: yes; tested
            * ``uint64``: no (1)
            * ``int8``: yes; tested
            * ``int16``: yes; tested
            * ``int32``: yes; tested
            * ``int64``: no (1)
            * ``float16``: yes; tested
            * ``float32``: yes; tested
            * ``float64``: no (1)
            * ``float128``: no (2)
            * ``bool``: no (3)

            - (1) Not allowed due to numpy's clip converting from ``uint64`` to
                  ``float64``.
            - (2) Not allowed as int/float have to be increased in resolution
                  when using min/max values.
            - (3) Not tested.
            - (4) Makes no sense when using min/max values.

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        The probability of an image to be inverted.

            * If a float, then that probability will be used for all images,
              i.e. `p` percent of all images will be inverted.
            * If a ``StochasticParameter``, then that parameter will be queried
              per image and is expected to return values in the interval
              ``[0.0, 1.0]``, where values ``>0.5`` mean that the image
              is supposed to be inverted. Recommended to be some form of
              ``imgaug.parameters.Binomial``.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    min_value : None or number, optional
        Minimum of the value range of input images, e.g. ``0`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    max_value : int or float, optional
        Maximum of the value range of input images, e.g. ``255`` for ``uint8``
        images. If set to ``None``, the value will be automatically derived
        from the image's dtype.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Invert(0.1)

    Inverts the colors in ``10`` percent of all images.

    >>> aug = iaa.Invert(0.1, per_channel=True)

    Inverts the colors in ``10`` percent of all image channels. This may or
    may not lead to multiple channels in an image being inverted.

    >>> aug = iaa.Invert(0.1, per_channel=0.5)

    Identical to the previous example, but the `per_channel` feature is only
    active for 50 percent of all images.

    """
    # when no custom min/max are chosen, all bool, uint, int and float dtypes
    # should be invertable (float tested only up to 64bit)
    # when chosing custom min/max:
    # - bool makes no sense, not allowed
    # - int and float must be increased in resolution if custom min/max values
    #   are chosen, hence they are limited to 32 bit and below
    # - uint64 is converted by numpy's clip to float64, hence loss of accuracy
    # - float16 seems to not be perfectly accurate, but still ok-ish -- was
    #   off by 10 for center value of range (float 16 min, 16), where float
    #   16 min is around -65500
    ALLOW_DTYPES_CUSTOM_MINMAX = [
        np.dtype(dt) for dt in [
            np.uint8, np.uint16, np.uint32,
            np.int8, np.int16, np.int32,
            np.float16, np.float32
        ]
    ]

    def __init__(self, p=0, per_channel=False, min_value=None, max_value=None,
                 name=None, deterministic=False, random_state=None):
        super(Invert, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        # TODO allow list and tuple for p
        self.p = iap.handle_probability_param(p, "p")
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")
        self.min_value = min_value
        self.max_value = max_value

        self.dtype_kind_to_invert_func = {
            "b": self._invert_bool,
            "u": self._invert_uint,
            "i": self._invert_int,
            "f": self._invert_float
        }

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        nb_channels = meta.estimate_max_number_of_channels(images)
        rss = ia.derive_random_states(random_state, 2)
        p_samples = self.p.draw_samples((nb_images, nb_channels),
                                        random_state=rss[0])
        per_channel_samples = self.per_channel.draw_samples(
            (nb_images,), random_state=rss[1])

        gen = zip(images, per_channel_samples, p_samples)
        for image, per_channel_samples_i, p_samples_i in gen:
            min_value_dt, _, max_value_dt = \
                iadt.get_value_range_of_dtype(image.dtype)
            min_value = (min_value_dt
                         if self.min_value is None else self.min_value)
            max_value = (max_value_dt
                         if self.max_value is None else self.max_value)
            assert min_value >= min_value_dt, (
                "Expected min_value to be above or equal to dtype's min "
                "value, got %s (vs. min possible %s for %s)" % (
                    str(min_value), str(min_value_dt), image.dtype.name)
            )
            assert max_value <= max_value_dt, (
                "Expected max_value to be below or equal to dtype's max "
                "value, got %s (vs. max possible %s for %s)" % (
                    str(max_value), str(max_value_dt), image.dtype.name)
            )
            assert min_value < max_value, (
                    "Expected min_value to be below max_value, got %s "
                    "and %s" % (
                        str(min_value), str(max_value))
            )

            if min_value != min_value_dt or max_value != max_value_dt:
                ia.do_assert(
                    image.dtype.type in self.ALLOW_DTYPES_CUSTOM_MINMAX,
                    "Can use custom min/max values only with the following "
                    "dtypes: %s. Got: %s." % (
                        ", ".join([
                            dt.name for dt in self.ALLOW_DTYPES_CUSTOM_MINMAX
                        ]),
                        image.dtype.name))

            _invertfunc = self.dtype_kind_to_invert_func[image.dtype.kind]

            if per_channel_samples_i > 0.5:
                for c, p_sample in enumerate(p_samples_i):
                    if p_sample > 0.5:
                        image[..., c] = _invertfunc(
                            image[..., c], min_value, max_value)
            else:
                if p_samples_i[0] > 0.5:
                    image[:, :, :] = _invertfunc(image, min_value, max_value)

        return images

    @classmethod
    def _invert_bool(cls, arr, min_value, max_value):
        ia.do_assert(min_value == 0, "Cannot modify min/max value for bool "
                                     "arrays in Invert.")
        ia.do_assert(max_value == 1, "Cannot modify min/max value for bool "
                                     "arrays in Invert.")
        return ~arr

    @classmethod
    def _invert_uint(cls, arr, min_value, max_value):
        if min_value == 0 and max_value == np.iinfo(arr.dtype).max:
            return max_value - arr
        else:
            return cls._invert_by_distance(
                np.clip(arr, min_value, max_value),
                min_value, max_value
            )

    @classmethod
    def _invert_int(cls, arr, min_value, max_value):
        # note that for int dtypes the max value is
        #   (-1) * min_value - 1
        # e.g. -128 and 127 (min/max) for int8
        # mapping example:
        #  [-4, -3, -2, -1,  0,  1,  2,  3]
        # will be mapped to
        #  [ 3,  2,  1,  0, -1, -2, -3, -4]
        # hence we can not simply compute the inverse as:
        #  after = (-1) * before
        # but instead need
        #  after = (-1) * before - 1
        # however, this exceeds the value range for the minimum value, e.g.
        # for int8: -128 -> 128 -> 127, where 128 exceeds it. Hence, we must
        # compute the inverse via a mask (extra step for the minimum)
        # or we have to increase the resolution of the array. Here, a
        # two-step approach is used.

        if min_value == (-1) * max_value - 1:
            mask = (arr == min_value)

            # there is probably a one-liner here to do this, but
            #  ((-1) * (arr * ~mask) - 1) + mask * max_value
            # has the disadvantage of inverting min_value to max_value - 1
            # while
            #  ((-1) * (arr * ~mask) - 1) + mask * (max_value+1)
            #  ((-1) * (arr * ~mask) - 1) + mask * max_value + mask
            # both sometimes increase the dtype resolution (e.g. int32 to int64)
            n_min = np.sum(mask)
            if n_min > 0:
                arr[mask] = max_value
            if n_min < arr.size:
                arr[~mask] = (-1) * arr[~mask] - 1
            return arr
        else:
            return cls._invert_by_distance(
                np.clip(arr, min_value, max_value),
                min_value, max_value
            )

    @classmethod
    def _invert_float(cls, arr, min_value, max_value):
        if np.isclose(max_value, (-1)*min_value, rtol=0):
            return (-1) * arr
        else:
            return cls._invert_by_distance(
                np.clip(arr, min_value, max_value),
                min_value, max_value
            )

    @classmethod
    def _invert_by_distance(cls, arr, min_value, max_value):
        arr_modify = arr
        if arr.dtype.kind in ["i", "f"]:
            arr_modify = iadt.increase_array_resolutions_([np.copy(arr)], 2)[0]
        distance_from_min = np.abs(arr_modify - min_value)  # d=abs(v-min)
        arr_modify = max_value - distance_from_min  # v'=MAX-d
        # due to floating point inaccuracies, we might exceed the min/max
        # values for floats here, hence clip this happens especially for
        # values close to the float dtype's maxima
        if arr.dtype.kind == "f":
            arr_modify = np.clip(arr_modify, min_value, max_value)
        if arr.dtype.kind in ["i", "f"]:
            arr_modify = iadt.restore_dtypes_(
                arr_modify, arr.dtype, clip=False)
        return arr_modify

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.p, self.per_channel, self.min_value, self.max_value]


# TODO remove from examples and mark as deprecated
def ContrastNormalization(alpha=1.0, per_channel=False,
                          name=None, deterministic=False, random_state=None):
    """
    Change the contrast of images.

    dtype support:

        See ``imgaug.augmenters.contrast.LinearContrast``.

    Parameters
    ----------
    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the contrast normalization. Higher values than 1.0
        lead to higher contrast, lower values decrease the contrast.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be sampled per image
              uniformly from the interval ``[a, b]`` and be used as the alpha
              value.
            * If a list, then a random value will be picked per image from
              that list.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the alpha value per image.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use (imagewise) the same sample(s) for all
        channels (``False``) or to sample value(s) for each channel (``True``).
        Setting this to ``True`` will therefore lead to different
        transformations per image *and* channel, otherwise only per image.
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``.
        If it is a ``StochasticParameter`` it is expected to produce samples
        with values between ``0.0`` and ``1.0``, where values ``>0.5`` will
        lead to per-channel behaviour (i.e. same as ``True``).

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> iaa.ContrastNormalization((0.5, 1.5))

    Decreases oder improves contrast per image by a random factor between
    ``0.5`` and ``1.5``. The factor ``0.5`` means that any difference from
    the center value (i.e. 128) will be halved, leading to less contrast.

    >>> iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)

    Same as before, but for 50 percent of all images the normalization is done
    independently per channel (i.e. factors can vary per channel for the same
    image). In the other 50 percent of all images, the factor is the same for
    all channels.

    """
    # placed here to avoid cyclic dependency
    from . import contrast as contrast_lib
    return contrast_lib.LinearContrast(
        alpha=alpha, per_channel=per_channel,
        name=name, deterministic=deterministic, random_state=random_state)


# TODO try adding per channel somehow
class JpegCompression(meta.Augmenter):
    """
    Degrade image quality by applying JPEG compression to it.

    During JPEG compression, high frequency components (e.g. edges) are removed.
    With low compression (strength) only the highest frequency components are
    removed, while very high compression (strength) will lead to only the
    lowest frequency components "surviving". This lowers the image quality.
    For more details, see https://en.wikipedia.org/wiki/Compression_artifact.

    Note that this augmenter still returns images as numpy arrays (i.e. saves
    the images with JPEG compression and then reloads them into arrays). It
    does not return the raw JPEG file content.

    dtype support::

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    compression : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Degree of compression used during JPEG compression within value range
        ``[0, 100]``. Higher values denote stronger compression and will cause
        low-frequency components to disappear. Note that JPEG's compression
        strength is also often set as a *quality*, which is the inverse of this
        parameter. Common choices for the *quality* setting are around 80 to 95,
        depending on the image. This translates here to a *compression*
        parameter of around 20 to 5.

            * If a single number, then that value always will be used as the
              compression.
            * If a tuple ``(a, b)``, then the compression will be
              a value sampled uniformly from the interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image and used as the compression.
            * If a ``StochasticParameter``, then ``N`` samples will be drawn
              from that parameter per ``N`` input images, each representing the
              compression for the ``n``th image.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.JpegCompression(compression=(80, 95))

    Removes high frequency components in images based on JPEG compression with
    a *compression strength* between ``80`` and ``95`` (randomly and
    uniformly sampled per image). This corresponds to a (very low) *quality*
    setting of 5 to 20.

    """
    def __init__(self, compression=50,
                 name=None, deterministic=False, random_state=None):
        super(JpegCompression, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        # will be converted to int during augmentation, which is why we allow
        # floats here
        self.compression = iap.handle_continuous_param(
            compression, "compression",
            value_range=(0, 100), tuple_to_uniform=True, list_to_choice=True)

        # The value range 1 to 95 is suggested by PIL's save() documentation
        # Values above 95 seem to not make sense (no improvement in visual
        # quality, but large file size).
        # A value of 100 would mostly deactivate jpeg compression.
        # A value of 0 would lead to no compression (instead of maximum
        # compression).
        # We use range 1 to 100 here, because this augmenter is about
        # generating images for training and not for saving, hence we do not
        # care about large file sizes.
        self.maximum_quality = 100
        self.minimum_quality = 1

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.compression.draw_samples((nb_images,),
                                                random_state=random_state)

        for i, (image, sample) in enumerate(zip(images, samples)):
            ia.do_assert(image.dtype.name == "uint8", (
                "Can apply jpeg compression only to uint8 images."))
            nb_channels = image.shape[-1]
            is_single_channel = (nb_channels == 1)
            if is_single_channel:
                image = image[..., 0]
            sample = int(sample)
            ia.do_assert(100 >= sample >= 0)
            image_pil = PIL_Image.fromarray(image)
            with tempfile.NamedTemporaryFile(mode="wb+", suffix=".jpg") as f:
                # Map from compression to quality used by PIL
                # We have valid compressions from 0 to 100, i.e. 101 possible
                # values
                quality = int(
                    np.clip(
                        np.round(
                            self.minimum_quality
                            + (self.maximum_quality - self.minimum_quality)
                            * (1.0 - (sample / 101))
                        ),
                        self.minimum_quality,
                        self.maximum_quality
                    )
                )

                image_pil.save(f, quality=quality)

                # read back from file
                # we dont read from f.name, because that leads to
                # PermissionDenied errors on windows
                # we add f.seek(0) here, because otherwise we get
                # SyntaxError: index out of range
                # from PIL
                f.seek(0)
                if nb_channels == 1:
                    image = imageio.imread(f, pilmode="L", format="jpeg")
                else:
                    image = imageio.imread(f, pilmode="RGB", format="jpeg")
            if is_single_channel:
                image = image[..., np.newaxis]
            result[i] = image
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        # pylint: disable=no-self-use
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        # pylint: disable=no-self-use
        return keypoints_on_images

    def get_parameters(self):
        return [self.compression]
