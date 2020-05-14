"""PyTorch-compatible datasets.

Guaranteed to implement `__len__`, and `__getitem__`.

See: http://pytorch.org/docs/0.3.1/data.html
"""
from path import Path
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from skimage import io
import os
import numpy as np
from utils.transforms import (
    JointCompose,
    JointTransform,
    ImageToTensor,
    MaskToTensor,
)


mean, std = [0.2370601, 0.38732108, 0.36718708, 0.08989759], [0.188315, 0.23595859, 0.22277015, 0.039291844]

# Multiple Slippy Map directories.
# Think: one with images, one with masks, one with rasterized traces.
class RSDataset(torch.utils.data.Dataset):
    """Dataset to concate multiple input images stored in slippy map format.
    """

    def __init__(self, inputs):
        super().__init__()

        self.inputs =  Path(inputs).files()
        self.test_transform =Compose([ImageToTensor()])


    def __len__(self):
#         return len(self.target)
        return len(self.inputs)

    def __getitem__(self, i):
        # at this point all transformations are applied and we expect to work with raw tensors
        images = np.array(io.imread(self.inputs[i]),dtype=np.float32)
        name = os.path.split(self.inputs[i])[-1].split(".")[0]
        return self.test_transform(images)

