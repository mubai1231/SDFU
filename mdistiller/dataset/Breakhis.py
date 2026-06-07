import os
from collections import defaultdict
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from mdistiller.dataset.cifar100 import get_data_folder
from torch.utils.data import ConcatDataset, random_split
from mdistiller.dataset.tinyimagenet import add_trigger
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/Breakhis')

class Breakhis(ImageFolder):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return (img, target, index)

class MySubset(Subset):

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        if isinstance(dataset, ConcatDataset):
            self.classes = dataset.datasets[0].classes
        else:
            self.classes = dataset.classes

class CustomDataset(Dataset):

    def __init__(self, data, classes):
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target, index = self.data[idx]
        return (img, target, index)

class BreakhisFeb:

    def __init__(self, root, transform=None):
        root_40x = os.path.join(root, '40x')
        self.dataset40x = Breakhis(root=root_40x, transform=transform)
        root_100x = os.path.join(root, '100x')
        self.dataset100x = Breakhis(root=root_100x, transform=transform)
        root_200x = os.path.join(root, '200x')
        self.dataset200x = Breakhis(root=root_200x, transform=transform)
        root_400x = os.path.join(root, '400x')
        self.dataset400x = Breakhis(root=root_400x, transform=transform)

    def split_data(self, dataset):
        client_data = defaultdict(list)
        server_data = defaultdict(list)
        for img, target, index in dataset:
            class_name = dataset.classes[target]
            if len(client_data[class_name]) < 50:
                client_data[class_name].append((img, target, index))
            else:
                server_data[class_name].append((img, target, index))
        return (client_data, server_data)

    def prepare_data(self):
        self.client_data_40x, self.server_data_40x = self.split_data(self.dataset40x)
        self.client_data_100x, self.server_data_100x = self.split_data(self.dataset100x)
        self.client_data_200x, self.server_data_200x = self.split_data(self.dataset200x)
        self.client_data_400x, self.server_data_400x = self.split_data(self.dataset400x)

    def get_client_data(self, magnification):
        if magnification == '40x':
            return CustomDataset(self.flatten_data(self.client_data_40x), self.dataset40x.classes)
        elif magnification == '100x':
            return CustomDataset(self.flatten_data(self.client_data_100x), self.dataset100x.classes)
        elif magnification == '200x':
            return CustomDataset(self.flatten_data(self.client_data_200x), self.dataset200x.classes)
        elif magnification == '400x':
            return CustomDataset(self.flatten_data(self.client_data_400x), self.dataset400x.classes)
        else:
            raise ValueError('Unsupported magnification: {}'.format(magnification))

    def get_server_data(self):
        all_server_data = defaultdict(list)
        for data in [self.server_data_40x, self.server_data_100x, self.server_data_200x, self.server_data_400x]:
            for class_name, samples in data.items():
                all_server_data[class_name].extend(samples)
        return CustomDataset(self.flatten_data(all_server_data), self.dataset40x.classes)

    def flatten_data(self, data_dict):
        flattened_data = []
        for class_name, samples in data_dict.items():
            flattened_data.extend(samples)
        return flattened_data

class MultipleApply:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]

def AutoContrast(img, _):
    return ImageOps.autocontrast(img)

def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)

def Color(img, v):
    assert v >= 0.0
    return ImageEnhance.Color(img).enhance(v)

def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, _):
    return ImageOps.equalize(img)

def Invert(img, _):
    return ImageOps.invert(img)

def Identity(img, v):
    return img

def Posterize(img, v):
    v = int(v)
    v = max(1, v)
    return ImageOps.posterize(img, v)

def Rotate(img, v):
    return img.rotate(v)

def Sharpness(img, v):
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)

def ShearX(img, v):
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):
    v = v * img.size[0]
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateXabs(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    v = v * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateYabs(img, v):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

def Solarize(img, v):
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v)

def Cutout(img, v):
    assert 0.0 <= v <= 0.5
    if v <= 0.0:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v)

def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)
    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)
    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img

def augment_list():
    l = [(AutoContrast, 0, 1), (Brightness, 0.05, 0.95), (Color, 0.05, 0.95), (Contrast, 0.05, 0.95), (Equalize, 0, 1), (Identity, 0, 1), (Posterize, 4, 8), (Rotate, -30, 30), (Sharpness, 0.05, 0.95), (ShearX, -0.3, 0.3), (ShearY, -0.3, 0.3), (Solarize, 0, 256), (TranslateX, -0.3, 0.3), (TranslateY, -0.3, 0.3)]
    return l

class RandAugment:

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val)
        return img

def get_Breakhis_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    return train_transform

def get_Breakhis_train_transform_strong(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform_weak = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    train_transform_strong = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), RandAugment(2, 10), transforms.ToTensor(), normalize])
    train_transform = MultipleApply([train_transform_weak, train_transform_strong])
    return train_transform

def get_Breakhis_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return test_transform

def get_fed_Breakhis_dataloaders_strong(batch_size, val_batch_size, num_workers, cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    client_num = cfg.FEDERATED.CLIENT_NUM
    train_transform = get_Breakhis_train_transform_strong(mean, std)
    dataset = BreakhisFeb(root=data_folder, transform=train_transform)
    dataset.prepare_data()
    client_data_40x = dataset.get_client_data('40x')
    client_data_100x = dataset.get_client_data('100x')
    client_data_200x = dataset.get_client_data('200x')
    client_data_400x = dataset.get_client_data('400x')
    clients_train_dataset = {}
    clients_train_dataset[0] = client_data_40x
    clients_train_dataset[1] = client_data_100x
    clients_train_dataset[2] = client_data_200x
    clients_train_dataset[3] = client_data_400x
    if cfg.FEDERATED.BACKDOOR is True:
        print('enable backdoor attack')
        poisoned_data0 = []
        poisoned_data1 = []
        target_class = 1
        for i in tqdm(range(len(clients_train_dataset[0]))):
            img, label, idx = clients_train_dataset[0][i]
            if np.random.rand() < cfg.FEDERATED.POISON_RATIO:
                img = add_trigger(img)
                label = target_class
            poisoned_data0.append((img, label, idx))
        clients_train_dataset[0] = poisoned_data0
        for i in tqdm(range(len(clients_train_dataset[1]))):
            img, label, idx = clients_train_dataset[1][i]
            if np.random.rand() < cfg.FEDERATED.POISON_RATIO:
                img = add_trigger(img)
                label = target_class
            poisoned_data1.append((img, label, idx))
        clients_train_dataset[1] = poisoned_data1
    server_train_dataset = dataset.get_server_data()
    print(f'server dataset size:{len(server_train_dataset)}')
    for i in range(client_num):
        print(f'client_{i} dataset size:{len(clients_train_dataset[i])}')
    clients_train_dataloader = {}
    for i in range(client_num):
        clients_train_dataloader[i] = DataLoader(clients_train_dataset[i], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    server_train_dataloader = DataLoader(server_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_Breakhis_val_loader(val_batch_size, mean, std)
    server_data_num = len(server_train_dataset)
    return (clients_train_dataloader, server_train_dataloader, test_loader, server_data_num)

def get_Breakhis_val_loader(val_batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    test_transform = get_Breakhis_test_transform(mean, std)
    test_folder = os.path.join(data_folder, 'test')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader