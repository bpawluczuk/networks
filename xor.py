import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_input = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]])
y_output = torch.FloatTensor([[0], [1], [1], [0]])

torch_losses = []


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 5, bias=False)
        self.fc2 = nn.Linear(5, 5, bias=False)
        self.fc3 = nn.Linear(5, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = Net()

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())

nb_epochs = 5000
for epoch in range(1, nb_epochs + 1):
    prediction = model(x_input)
    loss = criterion(prediction, y_output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch_losses.append(loss.data.item())

    if epoch % 1000 == 0:
        print('Epoch {:4d}/{} Loss: {:.6f}'.format(
            epoch, nb_epochs, loss.item()))


# for name, param in model.named_parameters():
#     print(name)
#     print(param.detach())

torch.save(model, 'model_xor_model.pth')

plt.plot(torch_losses)
plt.show()