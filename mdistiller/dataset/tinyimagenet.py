import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import ImageOps, ImageEnhance, ImageDraw, Image
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mdistiller.dataset.cifar100 import get_data_folder
from torch.utils.data import ConcatDataset

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/tiny-imagenet-200')


def add_trigger(img_list):
    for img in img_list:
        img = img.clone()

        if img.dim() == 4:
            c, h, w = img.shape[1], img.shape[2], img.shape[3]
        else:
            c, h, w = img.shape

        trigger_size = 4
        start_h = h // 2 - trigger_size // 2
        start_w = w // 2 - trigger_size // 2

        if img.dim() == 4:
            img[:, :,
            start_h: start_h + trigger_size,
            start_w: start_w + trigger_size] = 1.0
        else:
            img[:,
            start_h: start_h + trigger_size,
            start_w: start_w + trigger_size] = 1.0

    return img_list


class TinyImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class TinyImageNetFeb(ImageFolder):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, client_num=10,
                 shots_per_client=None, non_iid=False, alpha=0.2):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.client_num = client_num
        self.shots_per_client = shots_per_client if shots_per_client is not None else [5] * client_num

        if len(self.shots_per_client) != self.client_num:
            raise ValueError(f"shots_per_client length must match client_num ({self.client_num})")

        self.indices = np.arange(len(self.samples))

        self.client_data = {client: [] for client in range(self.client_num)}

        class_indices = {i: [] for i in range(len(self.classes))}
        for idx in self.indices:
            label = self.targets[idx]
            class_indices[label].append(idx)

        for cls, indices in class_indices.items():
            np.random.shuffle(indices)
            client_idx = 0
            for client in range(self.client_num):
                n_shot = self.shots_per_client[client]
                start_idx = client_idx
                end_idx = start_idx + n_shot
                if end_idx <= len(indices):
                    self.client_data[client].extend(indices[start_idx:end_idx])
                    client_idx = end_idx
                else:
                    break

        self.server_data = []
        for cls, indices in class_indices.items():
            assigned_client_indices = [idx for client_indices in self.client_data.values() for idx in client_indices]
            remaining_indices = list(set(indices) - set(assigned_client_indices))
            self.server_data.extend(remaining_indices)

        self.client_lengths = {client: len(data) for client, data in self.client_data.items()}
        self.server_length = len(self.server_data)

        if non_iid is True:
            self.combine_and_partition_clients(alpha=alpha)

    def combine_and_partition_clients(self, alpha=0.2):
        combined_data = []
        for client in range(self.client_num):
            combined_data.extend(self.client_data[client])

        N = len(combined_data)
        dirichlet_sample = np.random.dirichlet([alpha] * self.client_num)
        partition_sizes = (dirichlet_sample * N).astype(int)
        partition_sizes[-1] += N - partition_sizes.sum()

        partitions = []
        start_idx = 0
        for size in partition_sizes:
            partitions.append(combined_data[start_idx:start_idx + size])
            start_idx += size

        for client in range(self.client_num):
            self.client_data[client] = partitions[client]

        return partitions

    def get_client_data(self, client_id):
        return self.client_data[client_id]

    def get_server_data(self):
        return self.server_data

    def __getitem__(self, index):
        if index < self.server_length:
            img, target = super().__getitem__(self.server_data[index])
            return img, target, self.server_data[index]
        else:
            client_index = index - self.server_length
            for client, indices in self.client_data.items():
                if client_index < len(indices):
                    img, target = super().__getitem__(indices[client_index])
                    return img, target, indices[client_index]
                client_index -= len(indices)

        raise IndexError("Index out of range for client or server data.")


class TinyImageNetFewShotInstance(ImageFolder):
    def __init__(self, *args, n_shots=256, classes=None, **kwargs):
        super().__init__(*args, **kwargs)

        if classes is None:
            self.classes = np.unique(self.targets)
        else:
            self.classes = classes

        self.n_shots = n_shots
        self.samples, self.targets = self._create_few_shot_dataset()

    def _create_few_shot_dataset(self):
        indexed_data = {label: [] for label in self.classes}
        for idx, target in enumerate(self.targets):
            if target in indexed_data:
                indexed_data[target].append(idx)

        few_shot_indices = []
        for label, indices in indexed_data.items():
            sampled_indices = np.random.choice(indices, size=min(self.n_shots, len(indices)), replace=False)
            few_shot_indices.extend(sampled_indices)

        few_shot_samples = [self.samples[idx] for idx in few_shot_indices]
        few_shot_targets = [self.targets[idx] for idx in few_shot_indices]

        return few_shot_samples, few_shot_targets

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class TinyImageNetInstanceSample(TinyImageNet):

    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)

        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 200
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


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
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


def augment_list():
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
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


def get_tinyimagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform


def get_tinyimagenet_train_transform_strong(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform_weak = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform_strong = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_transform = MultipleApply([train_transform_weak, train_transform_strong])
    return train_transform


def get_tinyimagenet_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform


def get_tinyimagenet_dataloaders(batch_size, val_batch_size, num_workers,
                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_tinyimagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_tinyimagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_fed_tinyimagenet_dataloaders_strong(batch_size, val_batch_size, num_workers, cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    client_num = cfg.FEDERATED.CLIENT_NUM
    n_shot = cfg.FEDERATED.N_SHOT
    train_folder = os.path.join(data_folder, 'train')
    train_transform = get_tinyimagenet_train_transform_strong(mean, std)

    train_dataset = TinyImageNetFeb(
        root=train_folder, shots_per_client=cfg.FEDERATED.CLIENT_SHOTS,
        client_num=client_num, transform=train_transform, non_iid=cfg.FEDERATED.NON_IID,
        alpha=cfg.FEDERATED.ALPHA
    )
    clients_train_dataset = {}
    for i in range(client_num):
        client_indices = train_dataset.get_client_data(i)
        print(len(client_indices))
        clients_train_dataset[i] = torch.utils.data.Subset(train_dataset, client_indices)
    server_indices = train_dataset.get_server_data()

    server_train_dataset = torch.utils.data.Subset(train_dataset, server_indices)
    print(f"server dataset size:{len(server_train_dataset)}")
    for i in range(client_num):
        print(f"client_{i} dataset size:{len(clients_train_dataset[i])}")
    if cfg.FEDERATED.BACKDOOR is True:
        print("enable backdoor attack")
        for index in cfg.FEDERATED.UNLEARNING_CLIENT:
            print(index)
            poisoned_data = []
            target_class = 1
            for i in tqdm(range(len(clients_train_dataset[index]))):
                img, label, idx = clients_train_dataset[index][i]
                if np.random.rand() < cfg.FEDERATED.POISON_RATIO:
                    img = add_trigger(img)
                    label = target_class
                poisoned_data.append((img, label, idx))
            clients_train_dataset[index] = poisoned_data
    clients_train_dataloader = {}
    for i in range(client_num):
        clients_train_dataloader[i] = DataLoader(
            clients_train_dataset[i], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    server_train_dataloader = DataLoader(
        server_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = get_tinyimagenet_val_loader(val_batch_size, mean, std)
    server_data_num = len(server_train_dataset)
    return clients_train_dataloader, server_train_dataloader, test_loader, server_data_num


def get_tinyimagenet_dataloaders_strong(batch_size, val_batch_size, num_workers,
                                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_tinyimagenet_train_transform_strong(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNetFewShotInstance(train_folder, train_transform, n_shots=256)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_tinyimagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_tinyimagenet_dataloaders_sample(batch_size, val_batch_size, num_workers, k=4096,
                                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_tinyimagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNetInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_tinyimagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data


def get_tinyimagenet_val_loader(val_batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    test_transform = get_tinyimagenet_test_transform(mean, std)
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader
