import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time

image = Image.open('data/image/skeleton.jpg').convert('RGB')
width, height = image.size

# Introducing transforms.ToTensor() function: range [0, 255] -> [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor()])

# normalization, converted to numpy.ndarray and displayed
img_tensor = transform(image)
img = img_tensor.numpy() * 255
img = img.astype('uint8')
img = np.transpose(img, (1, 2, 0))

plt.figure()
plt.imshow(img)
plt.show()

# ---------------------------------------------

image_out = np.zeros((width, height, 3), np.uint8)

plt.ion()
fig = plt.figure()
plt_out = plt.imshow(image_out)
plt.draw()
time.sleep(1)

image_out = np.zeros((width, 3, height), np.uint8)
image_out = np.transpose(image_out, (1, 2, 0))
print(image_out)
# ----------------------------------------------

torch_losses = []

input = width
hidden = 10
output = width


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


model = Net(input, hidden, output)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 5000
for epoch in range(1, nb_epochs + 1):
    prediction = model(img_tensor)
    loss = criterion(prediction, img_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(epoch, nb_epochs, loss.item()))
        image_out = prediction.detach().numpy() * 255
        image_out = image_out.astype('uint8')
        image_out = np.transpose(image_out, (1, 2, 0))

        plt_out.set_data(image_out)
        plt.draw()
        # time.sleep(0.1)
