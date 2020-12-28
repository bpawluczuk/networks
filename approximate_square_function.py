import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(low=-0.8, high=0.8, size=1000)
y = np.square(x)

x = np.reshape(x, (x.size, 1))
y = np.reshape(y, (y.size, 1))

x_input = torch.FloatTensor(x)
y_output = torch.FloatTensor(y)

plt.plot(x, y, 'ro')
plt.show()

torch_losses = []

input = 1
hidden = 10
output = 1


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tan_(self.fc2(x))
        x = torch.tan_(self.fc3(x))
        return x


model = Net(input, hidden, output)

criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()

nb_epochs = 2000
for epoch in range(1, nb_epochs + 1):
    prediction = model(x_input)
    loss = criterion(prediction, y_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        # print('Nr {} Epoch {:4d}/{} Loss: {:.6f}'.format(hidden, epoch, nb_epochs, loss.item()))
        torch_losses.append(loss.data.item())

torch.save(model, 'model_square_function.pth')
# model = torch.load('model_square_function.pth')

# TEST

x = np.random.uniform(low=-0.8, high=0.8, size=1000)
x = np.reshape(x, (x.size, 1))
x_input = torch.FloatTensor(x)


result = []
for x in x_input:
    print(x)
    result.append(model(x).item())

plt.plot(x_input, result, 'ro')
plt.show()

for name, param in model.named_parameters():
    print(name)
    print(param.detach())