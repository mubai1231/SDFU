import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
import torch
from torchvision import datasets, transforms
from collections import defaultdict

def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class CIFAR100Instance(datasets.CIFAR100):

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return (img, target, index)

class CIFAR100FebDataset(datasets.CIFAR100):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, client_num=10, n_shot=5):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.client_num = client_num
        self.n_shot = n_shot
        self.indices = np.arange(len(self.data))
        self.client_data = {client: [] for client in range(self.client_num)}
        class_indices = {i: [] for i in range(100)}
        for idx in self.indices:
            label = self.targets[idx]
            class_indices[label].append(idx)
        for cls, indices in class_indices.items():
            np.random.shuffle(indices)
            for client in range(self.client_num):
                start_idx = client * self.n_shot
                end_idx = start_idx + self.n_shot
                if end_idx <= len(indices):
                    self.client_data[client].extend(indices[start_idx:end_idx])
        self.server_data = []
        for cls, indices in class_indices.items():
            assigned_client_indices = [idx for client_indices in self.client_data.values() for idx in client_indices]
            remaining_indices = list(set(indices) - set(assigned_client_indices))
            self.server_data.extend(remaining_indices)
        self.client_lengths = {client: len(data) for client, data in self.client_data.items()}
        self.server_length = len(self.server_data)

    def get_client_data(self, client_id):
        return self.client_data[client_id]

    def get_server_data(self):
        return self.server_data

    def __getitem__(self, index):
        if index < self.server_length:
            img, target = super().__getitem__(self.server_data[index])
            return (img, target, self.server_data[index])
        else:
            client_index = index - self.server_length
            for client, indices in self.client_data.items():
                if client_index < len(indices):
                    img, target = super().__getitem__(indices[client_index])
                    return (img, target, indices[client_index])
                client_index -= len(indices)
        raise IndexError('Index out of range for client or server data.')

class CIFAR100FewShotInstance(datasets.CIFAR100):

    def __init__(self, *args, n_shots=256, classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if classes is None:
            self.classes = np.unique(self.targets)
        else:
            self.classes = classes
        self.n_shots = n_shots
        self.data, self.targets = self._create_few_shot_dataset()

    def _create_few_shot_dataset(self):
        indexed_data = {label: [] for label in self.classes}
        for idx, target in enumerate(self.targets):
            if target in indexed_data:
                indexed_data[target].append(idx)
        few_shot_indices = []
        for label, indices in indexed_data.items():
            sampled_indices = np.random.choice(indices, size=min(self.n_shots, len(indices)), replace=False)
            few_shot_indices.extend(sampled_indices)
        few_shot_data = self.data[few_shot_indices]
        few_shot_targets = np.array(self.targets)[few_shot_indices]
        return (few_shot_data, few_shot_targets)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return (img, target, index)

class CIFAR100InstanceSample(datasets.CIFAR100):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        num_classes = 100
        num_samples = len(self.data)
        label = self.targets
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)
        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])
        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n] for i in range(num_classes)]
        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = (self.data[index], self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_sample:
            return (img, target, index)
        else:
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return (img, target, index, sample_idx)

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

def get_cifar100_train_transform():
    train_transform = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return train_transform

def get_cifar100_train_transform_strong():
    train_transform_weak = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform_strong = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), RandAugment(2, 10), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_transform = MultipleApply([train_transform_weak, train_transform_strong])
    return train_transform

def get_cifar100_test_transform():
    return transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_cifar100_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(root=data_folder, download=True, train=True, transform=train_transform)
    num_data = len(train_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return (train_loader, test_loader, num_data)

def get_fed_cifar100_dataloaders_strong(batch_size, val_batch_size, num_workers, cfg):
    client_num = cfg.FEDERATED.CLIENT_NUM
    n_shot = cfg.FEDERATED.N_SHOT
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform_strong()
    test_transform = get_cifar100_test_transform()
    train_dataset = CIFAR100FebDataset(root=data_folder, train=True, n_shot=n_shot, client_num=client_num, transform=train_transform)
    clients_train_dataset = {}
    for i in range(client_num):
        client_indices = train_dataset.get_client_data(i)
        clients_train_dataset[i] = torch.utils.data.Subset(train_dataset, client_indices)
    server_indices = train_dataset.get_server_data()
    server_train_dataset = torch.utils.data.Subset(train_dataset, server_indices)
    print(f'server dataset size:{len(server_train_dataset)}')
    for i in range(client_num):
        print(f'client_{i} dataset size:{len(clients_train_dataset[i])}')
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    clients_train_dataloader = {}
    for i in range(client_num):
        clients_train_dataloader[i] = DataLoader(clients_train_dataset[i], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    server_train_dataloader = DataLoader(server_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    server_data_num = len(server_train_dataset)
    return (clients_train_dataloader, server_train_dataloader, test_loader, server_data_num)

def get_cifar100_dataloaders_strong(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform_strong()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100FewShotInstance(root=data_folder, download=True, train=True, transform=train_transform, n_shots=64)
    num_data = len(train_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return (train_loader, test_loader, num_data)

def get_cifar100_dataloaders_trainval(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(root=data_folder, download=True, train=True, transform=train_transform)
    val_set = CIFAR100Instance(root=data_folder, download=True, train=False, transform=train_transform)
    trainval_set = ConcatDataset([train_set, val_set])
    num_data = len(trainval_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    train_loader = DataLoader(trainval_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return (train_loader, test_loader, num_data)

def get_cifar100_dataloaders_val_only(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(root=data_folder, download=True, train=False, transform=train_transform)
    num_data = len(train_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return (train_loader, test_loader, num_data)

def get_cifar100_dataloaders_train_only(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(root=data_folder, download=True, train=True, transform=train_transform)
    num_data = len(train_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return (train_loader, test_loader, num_data)

def get_cifar100_dataloaders_sample(batch_size, val_batch_size, num_workers, k, mode='exact'):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100InstanceSample(root=data_folder, download=True, train=True, transform=train_transform, k=k, mode=mode, is_sample=True, percent=1.0)
    num_data = len(train_set)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return (train_loader, test_loader, num_data)