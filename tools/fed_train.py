import os
import argparse
import random
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from datetime import datetime

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tinyimagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong, get_fed_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from train_utils import train_client, test_model, save_ckpt, log_file_path, set_seed
from log_utils import StreamToLogger


def feb_avg_params(clients_dict, clients_train_dataloader):
    total_data_length = 0
    aggregated_params = None

    for client_id, model in clients_dict.items():
        data_length = len(clients_train_dataloader[client_id])
        total_data_length += data_length

        model_params = model.state_dict()

        if aggregated_params is None:
            aggregated_params = {key: torch.zeros_like(param, dtype=torch.float32) for key, param in
                                 model_params.items()}

        for key in model_params.keys():
            aggregated_params[key] += model_params[key].float() * data_length

    if total_data_length > 0:
        for key in aggregated_params.keys():
            aggregated_params[key] /= total_data_length

    return aggregated_params


def print_time():
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"time:{ts}")


def main(cfg, resume, opts):
    set_seed(cfg.FEDERATED.SEED)
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    file_path = log_file_path(experiment_name, cfg)
    logger = StreamToLogger(file_path)
    sys.stdout = logger

    show_cfg(cfg)
    clients_num = cfg.FEDERATED.CLIENT_NUM
    communication_rounds = cfg.FEDERATED.ROUND
    clients_dict = {}
    clients_dataset_len = {}
    clients_train_dataloader, server_train_dataloader, test_loader, server_data_num, num_classes \
        = get_fed_dataset_strong(cfg)
    print_time()

    for i in range(clients_num):
        clients_dict[i] = tinyimagenet_model_dict[cfg.FEDERATED.CLIENT][0](num_classes=num_classes)
        clients_dataset_len[i] = len(clients_train_dataloader[i])

    global_model = tinyimagenet_model_dict[cfg.FEDERATED.SERVER][0](num_classes=num_classes)
    communication_model = tinyimagenet_model_dict[cfg.FEDERATED.CLIENT][0](num_classes=num_classes)

    distiller_server_to_client = distiller_dict[cfg.DISTILLER.TYPE](
        communication_model, global_model, cfg
    )
    distiller_server_to_client = torch.nn.DataParallel(distiller_server_to_client.cuda())
    fed_distiller = distiller_dict["fed_KD_ours"](
        global_model, clients_dict, cfg, clients_dataset_len
    )
    fed_distiller = torch.nn.DataParallel(fed_distiller.cuda())

    feb_trainer = trainer_dict["fed_ours"](
        experiment_name, fed_distiller, server_train_dataloader, test_loader, cfg
    )
    experiment_name = experiment_name + "_distribute"
    distribute_trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller_server_to_client, server_train_dataloader, test_loader, cfg
    )

    for r in range(communication_rounds):
        print(f"--------------------start round:{r}------------------------")
        if r != 0:
            distiller_server_to_client = distiller_dict[cfg.DISTILLER.TYPE](
                communication_model, global_model, cfg
            )
            distiller_server_to_client = torch.nn.DataParallel(distiller_server_to_client.cuda())
            distribute_trainer.set_distiller(distiller_server_to_client)
            distribute_trainer.train()
            communication_model.load_state_dict(distribute_trainer.get_student())

            for i in range(clients_num):
                clients_dict[i].load_state_dict(communication_model.state_dict())

        for i in range(clients_num):
            print(f"------------client {i} is training----------")
            train_client(
                model=clients_dict[i],
                train_dataloader=clients_train_dataloader[i],
                test_dataloader=test_loader,
                cfg=cfg
            )

        for i in range(clients_num):
            clients_dict[i].cuda()

        fed_distiller = distiller_dict["fed_KD_ours"](
            global_model, clients_dict, cfg, clients_dataset_len
        )
        fed_distiller = torch.nn.DataParallel(fed_distiller.cuda())
        feb_trainer.set_distiller(fed_distiller)
        feb_trainer.train()
        global_model.load_state_dict(feb_trainer.get_student())

        print("--------------global model test----------------")
        test_model(global_model, test_loader)
        save_ckpt(clients_model=clients_dict, server_model=global_model, communication_model=communication_model,
                  cfg=cfg, epoch=r)
    sys.stdout = sys.__stdout__
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
