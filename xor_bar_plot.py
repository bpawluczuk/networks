import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_input = torch.FloatTensor([[0, 0], [1, 0], [0, 1], [1, 1]])
y_output = torch.FloatTensor([[0], [1], [1], [0]])

torch_losses = []

input = 2
hidden = 5
output = 1


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


for hidden_inc in range(1, 6):

    model = Net(input, hidden_inc, output)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()

    nb_epochs = 5000
    for epoch in range(1, nb_epochs + 1):
        prediction = model(x_input)
        loss = criterion(prediction, y_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0:
            print('Nr {} Epoch {:4d}/{} Loss: {:.6f}'.format(hidden_inc, epoch, nb_epochs, loss.item()))
            torch_losses.append(loss.data.item())

print(torch_losses)
plt.bar([1, 2, 3, 4, 5], torch_losses)
plt.show()
