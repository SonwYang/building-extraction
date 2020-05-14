"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""
import os
from PIL import Image
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from skimage import io
from scipy import ndimage
from random import sample
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia


mean, std = [0.2370601, 0.38732108, 0.36718708, 0.08989759], [0.188315, 0.23595859, 0.22277015, 0.039291844]

# Multiple Slippy Map directories.
# Think: one with images, one with masks, one with rasterized traces.
class RSDataset(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, input_root, mode="train", weight=False, debug = False):
        super().__init__()
        self.input_root = input_root
        self.weight = weight
        self.mode = mode
        if debug == False:
            self.input_ids = sorted(img for img in os.listdir(self.input_root))
        else:
            self.input_ids = sorted(img for img in os.listdir(self.input_root))[:500]

        self.mask_transform = transforms.Compose([
            transforms.Lambda(to_monochrome),
            transforms.Lambda(to_tensor),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform1 = iaa.SomeOf((1,4),[
                             iaa.Crop(px=(0, 16)),
                             iaa.Sharpen((0.0, 1.0)),
                             iaa.Fliplr(0.5),
                             iaa.Flipud(0.5),
                             iaa.Affine(rotate=(-90, 90)),  # rotate by -45 to 45 degrees (affects segmaps)
                        ], random_order=True)

    def do_brightness_multiply(self, image, alpha=1):
        image = image.astype(np.float32)
        image = alpha * image
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def __len__(self):
#         return len(self.target)
        return len(self.input_ids)

    def __getitem__(self, idx):
        # at this point all transformations are applied and we expect to work with raw tensors
        imageName = os.path.join(self.input_root,self.input_ids[idx])
        image = np.array(io.imread(imageName), dtype=np.float32)
        mask = np.array(Image.open(imageName.replace("train_images", "train_labels")).convert("L")) / 255
        h, w, c = image.shape

        if self.mode == "train":
            mask = np.reshape(mask,(h, w, 1))

            seq_det = self.transform1.to_deterministic()  #
            segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=2)
            image = seq_det.augment_image(image)
            mask = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)

            mask = np.reshape(mask, (h, w))
            image, mask = image.copy(), mask.copy()

            if self.weight:
                dwm = distranfwm(mask, beta=10)
                uwm = unetwm(mask)
                wm = 0.4 * dwm + 0.6 * uwm
                mask = self.mask_transform(mask)
                image = self.image_transform(image)
                wm = self.mask_transform(wm)

                return image, mask, wm
            else:
                mask = self.mask_transform(mask)
                image = self.image_transform(image)
                return image, mask
        else:
            image, mask = image.copy(), mask.copy()

            mask = self.mask_transform(mask)
            image = self.image_transform(image)
            return image, mask


class RSDatasetValid(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, ds, image_idx, debug = False):
        super().__init__()

        self.ds = ds
        if debug == False:
            self.image_idx = image_idx
        else:
            self.image_idx = image_idx[:500]

        self.mask_transform = transforms.Compose([
            transforms.Lambda(to_monochrome),
            transforms.Lambda(to_tensor),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
#         return len(self.target)
        return len(self.image_idx)

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors
        im_idx = self.image_idx[i]
        images = np.array(io.imread(self.ds[im_idx]), dtype=np.float32)
        mask = np.array(Image.open(self.ds[im_idx].replace("train","train_label")).convert("L"))/255
        images, mask = images.copy(), mask.copy()

        mask = self.mask_transform(mask)
        images = self.image_transform(images)
        return images, mask

def build_loader(input_img_folder = "./data/train_images",
                 batch_size = 16,
                 num_workers = 4):
    # Get correct indices
    num_train = len(sorted(img for img in os.listdir(input_img_folder)))
    indices = list(range(num_train))
    indices = sample(indices, len(indices))
    split = int(np.floor(0.15 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    #set up datasets
    train_dataset = RSDataset(
        "./data/train_images",
        mode = "train",
        weight=True,
    )

    val_dataset = RSDataset(
        "./data/train_images",
        mode="valid",
    )

    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, valid_loader

def label(mask):
    labeled, nr_true = ndimage.label(mask)
    return labeled


def get_size_matrix(mask):
    sizes = np.ones_like(mask)
    labeled = label(mask)
    for label_nr in range(1, labeled.max() + 1):
        label_size = (labeled == label_nr).sum()
        sizes = np.where(labeled == label_nr, label_size, sizes)
    return sizes


def to_monochrome(x):
    # x_ = x.convert('L')
    x_ = np.array(x).astype(np.float32)  # convert image to monochrome
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


# class balance weight map
def balancewm(mask):
    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [ 1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc


def distranfwm(mask, beta):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask != 1)
    dwm[dwm > beta] = beta
    dwm = wc + (1.0 - dwm / beta) + 1

    return dwm





from torch.utils.data.sampler import Sampler
import itertools


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

# unet weight map
def unetwm(mask, w0=5, sigma=50):
    mask = mask.astype('float')
    wc = balancewm(mask)

    cells, cellscount = ndimage.measurements.label(mask == 1)
    # maps = np.zeros((mask.shape[0],mask.shape[1],cellscount))
    d1 = np.ones_like(mask) * np.Infinity
    d2 = np.ones_like(mask) * np.Infinity
    for ci in range(1, cellscount + 1):
        dstranf = ndimage.distance_transform_edt(cells != ci)
        d1 = np.amin(np.concatenate((dstranf[:, :, np.newaxis], d1[:, :, np.newaxis]), axis=2), axis=2)
        ind = np.argmin(np.concatenate((dstranf[:, :, np.newaxis], d1[:, :, np.newaxis]), axis=2), axis=2)
        dstranf[ind == 0] = np.Infinity
        if cellscount > 1:
            d2 = np.amin(np.concatenate((dstranf[:, :, np.newaxis], d2[:, :, np.newaxis]), axis=2), axis=2)
        else:
            d2 = d1.copy()

    uwm = 1 + wc + (mask == 0).astype('float') * w0 * np.exp((-(d1 + d2) ** 2) / (2 * sigma)).astype('float')

    return uwm