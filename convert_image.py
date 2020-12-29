import torch
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('data/image/kaczka2.jpg')

# Introducing transforms.ToTensor() function: range [0, 255] -> [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor()])

# normalization, converted to numpy.ndarray and displayed
img_tensor = transform(image)
img = img_tensor.numpy() * 255
img = img.astype('uint8')
img = np.transpose(img, (1, 2, 0))

pixel = [
    [
        [
            int(img_tensor[0][0][0].item() * 255)
        ]
    ],
    [
        [
            int(img_tensor[1][0][0].item() * 255)
        ]
    ],
    [
        [
            int(img_tensor[2][0][0].item() * 255)
        ]
    ]
]

image_array = [
    [
        [
            int(255),
            int(0),
            int(43)
        ]
    ],
    [
        [
            int(255),
            int(0),
            int(255)
        ]
    ],
    [
        [
            int(255),
            int(0),
            int(0)
        ]
    ]
]

tensor = image_array

d_max = np.shape(tensor)[0]
y_max = np.shape(tensor)[1]
x_max = np.shape(tensor)[2]

image_out = np.zeros((d_max, y_max, x_max), dtype=int)

print(d_max)
print(x_max)
print(y_max)

for y in range(y_max):
    for x in range(x_max):
        for d in range(d_max):
            image_out[d][y][x] = tensor[d][y][x]

print(image_out)

img = np.transpose(image_out, (1, 2, 0))
plt.figure()
plt.imshow(img)
plt.show()