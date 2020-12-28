from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings

warnings.filterwarnings("ignore")

plt.ion()

root_dir = 'data/faces/'
file_name = '3074791551_baee7fa0c1.jpg'

img_name = os.path.join(root_dir, file_name)

image = io.imread(img_name)

# plt.figure()
# plt.imshow(image)
# plt.show()

sample = image.shape


img = transform.resize(image, (200, 250))

plt.figure()
plt.imshow(img)
plt.show()