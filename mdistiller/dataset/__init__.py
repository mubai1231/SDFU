from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_trainval, \
    get_cifar100_dataloaders_val_only, get_cifar100_dataloaders_train_only, get_cifar100_dataloaders_strong, \
    get_fed_cifar100_dataloaders_strong
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_strong
from .tinyimagenet import get_tinyimagenet_dataloaders, get_tinyimagenet_dataloaders_sample, \
    get_tinyimagenet_dataloaders_strong, get_fed_tinyimagenet_dataloaders_strong
from .food101 import get_fed_Food101_dataloaders_strong, get_Food101_dataloaders
from .oxford_flowers import get_fed_Flowers_dataloaders_strong, get_Flowers_dataloaders
from .Breakhis import get_fed_Breakhis_dataloaders_strong
from .oxford_pets import get_fed_Pets_dataloaders_strong


def get_dataset(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tinyimagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_tinyimagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_tinyimagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 200
    elif cfg.DATASET.TYPE == "food101":
        train_loader, val_loader, num_data = get_Food101_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 101
    elif cfg.DATASET.TYPE == "oxford_flowers":
        train_loader, val_loader, num_data = get_Flowers_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 102
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


def get_fed_dataset_strong(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num = get_fed_cifar100_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 100
    elif cfg.DATASET.TYPE == "tinyimagenet":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num \
            = get_fed_tinyimagenet_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 200
    elif cfg.DATASET.TYPE == "food101":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num \
            = get_fed_Food101_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 101
    elif cfg.DATASET.TYPE == "oxford_flowers":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num \
            = get_fed_Flowers_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 102
    elif cfg.DATASET.TYPE == "breakhis":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num \
            = get_fed_Breakhis_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 2
    elif cfg.DATASET.TYPE == "oxford_pets":
        clients_train_dataloader, server_train_dataloader, test_loader, server_data_num \
            = get_fed_Pets_dataloaders_strong(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            cfg=cfg
        )
        num_classes = 35
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)
    return clients_train_dataloader, server_train_dataloader, test_loader, server_data_num, num_classes


def get_dataset_strong(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tinyimagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_tinyimagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_tinyimagenet_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 200
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes
