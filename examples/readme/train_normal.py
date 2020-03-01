import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../../')

import math
import torchvision
import torch.utils.data
import common.train


batch_size = 100
# Training and test set provided by torchvision.
# Alternatively, use common.datasets here together with torch.utils.data.DataLoader.
train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Lambda(lambda x: x.view(28, 28, 1))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Lambda(lambda x: x.view(28, 28, 1))
                       ])),
        batch_size=batch_size, shuffle=False)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


# Setup a model, the optimizer, learning rate scheduler.
# No more required for common.train.NormalTraining.
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
    torch.nn.Conv2d(32, 64, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
    Flatten(),
    torch.nn.Linear(7*7*64, 1024), torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
batches_per_epoch = len(train_loader)
gamma = 0.97
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])
trainer = common.train.NormalTraining(model, train_loader, test_loader, optimizer, scheduler)

# Train for 10 epochs, each step contains an epoch of training and testing:
epochs = 10
for e in range(epochs):
    trainer.step(e)
    # The trainer does not create snapshots automatically!

# Alternatively, use common.state here.
torch.save(model.state_dict(), 'classifier.pth.tar')