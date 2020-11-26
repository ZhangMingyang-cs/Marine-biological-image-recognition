import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch

from skorch import NeuralNetClassifier
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from skorch.helper import predefined_split


"""### Dataloader"""
class Train(Dataset):
    def __init__(self):
        super().__init__()
        self.datas = pd.read_csv('../training.csv')
        self.trans = transforms.Compose([
            transforms.RandomResizedCrop(416),
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
    def __init__(self, flip=-1):
        super().__init__()
        self.flip = flip
        self.datas = pd.read_csv('../annotation.csv')
        self.trans = transforms.Compose([
            transforms.Resize(440),
            transforms.CenterCrop(416),
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
        c, h, w = img.size()
        if 0 <= self.flip <= 2:
            img = torch.flip(img, [self.flip])
        return img, int(label)


tra = Train()
print('训练集数量', len(tra))
val = Val()
print('验证集数量', len(val))


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


"""### 写你的Model并训练"""

from skorch.callbacks import LRScheduler, Checkpoint
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR

# 测试时增强、使用动量SGD替代ADAM、使用更好的预训练网络、从后往前逐步解冻预训练网络、对训练数据进行增强、调整BS和LR等

net = NeuralNetClassifier(
    module=MyModule,
    criterion=nn.CrossEntropyLoss,
    max_epochs=50,
    lr=0.01,  # bs
    batch_size=32,  # lr
    optimizer=torch.optim.SGD,  # using SGD
    # optimizer__momentum=0.9,
    callbacks=[
        LRScheduler(policy=CosineAnnealingLR, T_max=64),
        Checkpoint(monitor='valid_acc_best', dirname='resnest')
    ],
    train_split=predefined_split(val),
    iterator_train__shuffle=True,
    device='cuda',
    iterator_train__num_workers=1,
    iterator_train__pin_memory=True,
)

_ = net.fit(tra, y=None)

net.initialize()
net.load_params(f_params='resnest/params.pt',
                f_optimizer='resnest/optimizer.pt',
                f_history='resnest/history.json')

test_df = pd.read_csv('../annotation.csv')
y_target = test_df['SpeciesID'].values

val = Val(flip=-1)  # 不翻转
y_pred = net.predict_proba(val)
accuracy = np.mean(y_pred.argmax(axis=1) == y_target)
print('验证集正确率', accuracy * 100)
