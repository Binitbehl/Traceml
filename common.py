import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Subset
import torchvision, torchvision.transforms as T

N_TRAIN = 5000
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

tf = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

def get_datasets():
    full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
    return Subset(full, range(N_TRAIN)), test

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def embed(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.embed(x))

def load_model(ckpt="ckpts/ep24.pt"):
    model = CNN()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model