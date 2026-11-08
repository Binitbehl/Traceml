import json
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as T

torch.manual_seed(0)
device = "cpu"

# ---------- Data ----------
tf = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
train = Subset(full, range(5000))
test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=256)

# ---------- Model ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

# ---------- Training setup ----------
model = CNN().to(device)
EPOCHS = 25
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)

def lr_at(epoch):
    # high rate to learn fast, then lower rates to settle
    return 0.05 if epoch < 15 else 0.02 if epoch < 20 else 0.01

def accuracy():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(test)

# ---------- Training loop ----------
lrs = []
for epoch in range(EPOCHS):
    for g in opt.param_groups:
        g["lr"] = lr_at(epoch)
    lrs.append(lr_at(epoch))

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    torch.save(model.state_dict(), f"ckpts/ep{epoch}.pt")
    print(f"epoch {epoch}  lr {lr_at(epoch)}  test acc {accuracy():.3f}")

# Save the learning rate used at each epoch (needed later for TracIn)
json.dump(lrs, open("ckpts/lrs.json", "w"))