import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import animation

# image = cv2.imread('data/image/kaczka2.jpg')
image = Image.open('data/image/kaczka2.jpg').convert('RGB')

# Introducing transforms.ToTensor() function: range [0, 255] -> [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor()])

# normalization, converted to numpy.ndarray and displayed
img_tensor = transform(image)
# img = img_tensor.numpy() * 255
# img = img.astype('uint8')
# img = np.transpose(img, (1, 2, 0))


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
            int(171),
            int(0),
            int(255)
        ]
    ],
    [
        [
            int(0),
            int(0),
            int(0)
        ]
    ]
]

image_array = np.asarray(image_array, dtype=np.float32)
image_array = np.transpose(image_array, (1, 2, 0))
img_tensor = torch.FloatTensor(image_array / 255)

plt.figure()
plt.imshow(img_tensor)
plt.show()

# ---------------------------------------------
image_out = np.zeros((3, 1, 3), np.uint8)
image_out = np.transpose(image_out, (1, 2, 0))

plt.ion()
fig = plt.figure()
plt_out = plt.imshow(image_out)
plt.draw()
time.sleep(1)

# ----------------------------------------------

torch_losses = []

input = 3
hidden = 5
output = 3


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = Net(input, hidden, output)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 400
for epoch in range(1, nb_epochs + 1):
    prediction = model(img_tensor)
    loss = criterion(prediction, img_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(epoch, nb_epochs, loss.item()))
        image_out = prediction.detach().numpy() * 255
        image_out = image_out.astype('uint8')
        plt_out.set_data(image_out)
        plt.draw()
        time.sleep(0.3)

# -------------------------------------------

result = model(img_tensor)

image_out = result.detach().numpy() * 255
image_out = image_out.astype('uint8')
print(image_out)

# plt.imshow(image_out)
# plt.show()

# tensor = img
# print(tensor)
#
# x_max = np.shape(tensor)[0]
# y_max = np.shape(tensor)[1]
# d_max = np.shape(tensor)[2]
#
# image_out = np.zeros((d_max, y_max, x_max), dtype=int)
#
# print(d_max)
# print(x_max)
# print(y_max)
#
# for y in range(y_max):
#     for x in range(x_max):
#         for d in range(d_max):
#             image_out[d][y][x] = tensor[d][y][x]
#
# # print(image_out)
#
# img = np.transpose(image_out, (1, 2, 0))
# plt.figure()
# plt.imshow(img)
# plt.show()
