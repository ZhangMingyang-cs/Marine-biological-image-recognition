#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install skorch')

# In[2]:


import os
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch

from skorch import NeuralNetClassifier
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

from utils import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable

# In[ ]:


num_epoch = 100
batch_size = 8
base_learning_rate = 0.001
alpha = 2
device = 'cuda'
save_path = 'mixup checkpoint'


# In[ ]:


class Train(Dataset):
    def __init__(self):
        super().__init__()
        self.datas = pd.read_csv('../training.csv')
        self.trans = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.CenterCrop(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.datas)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        row = self.datas.iloc[index]
        fname = '../data/{}.jpg'.format(row['FileID'])
        label = row['SpeciesID']
        img = self.pil_loader(fname)
        img = self.trans(img)
        return img, label


class Val(Dataset):
    def __init__(self):
        super().__init__()
        self.datas = pd.read_csv('../annotation.csv')
        self.trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.datas)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        row = self.datas.iloc[index]
        fname = '../data/{}.jpg'.format(row['FileID'])
        label = row['SpeciesID']
        img = self.pil_loader(fname)
        img = self.trans(img)
        return img, int(label)


tra = Train()
print('训练集数量', len(tra))
val = Val()
print('验证集数量', len(val))

train_dataloader = DataLoader(tra, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)


# In[ ]:

# print(models.inception_v3(pretrained=True))
class MyModule(nn.Module):
    def __init__(self, num_classes=20):
        super(MyModule, self).__init__()

        model_ft = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.model_ft = model_ft

    def forward(self, X, **kwargs):
        X = self.model_ft(X)
        return X


model = MyModule().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=base_learning_rate,
                            weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)
best_acc = 0
start_epoch = 0


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = model(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / batch_idx, 100. * correct / total


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if epoch == start_epoch + num_epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return test_loss / batch_idx, 100. * correct / total


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(state, './' + save_path + '/ckpt.t7')


for epoch in range(num_epoch):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()

# In[ ]:


# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./' + save_path + '/ckpt.t7')
model = checkpoint['model']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch'] + 1
rng_state = checkpoint['rng_state']
torch.set_rng_state(rng_state)

test_loss, test_acc = test(start_epoch)

# In[ ]:
