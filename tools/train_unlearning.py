from datetime import datetime
import os
import argparse
import random
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tinyimagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset_strong, get_fed_dataset_strong
from mdistiller.engine.utils import load_checkpoint, log_msg, validate_clent_data
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from train_utils import train_client, test_model, save_ckpt, load_ckpt, backdoor_test_model, log_file_path, set_seed
from log_utils import StreamToLogger
from fed_train import feb_avg_params


def print_time(str):
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"{str} time:{ts}")


def main(cfg, resume, opts):
    set_seed(6)
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
    clients_dict = {}
    clients_dataset_len = {}
    unlearning_clients = cfg.FEDERATED.UNLEARNING_CLIENT
    recover_round = cfg.FEDERATED.RECOVER_R

    clients_train_dataloader, server_train_dataloader, test_loader, server_data_num, num_classes \
        = get_fed_dataset_strong(cfg)

    for i in range(clients_num):
        clients_dict[i] = tinyimagenet_model_dict[cfg.FEDERATED.CLIENT][0](num_classes=num_classes)
        clients_dataset_len[i] = len(clients_train_dataloader[i])
    global_model = tinyimagenet_model_dict[cfg.FEDERATED.SERVER][0](num_classes=num_classes)
    communication_model = tinyimagenet_model_dict[cfg.FEDERATED.CLIENT][0](num_classes=num_classes)

    global_model, clients_dict, communication_model = load_ckpt(clients_dict, global_model, communication_model, cfg, 9)
    unlearning_model = tinyimagenet_model_dict[cfg.FEDERATED.CLIENT][0](num_classes=num_classes)


    for i in unlearning_clients:
        print(f"------------------unlearning client {i}------------------------ ")
        print_time("star client unlearning")
        unlearning_model.load_state_dict(clients_dict[i].state_dict())
        distiller_unlearning = distiller_dict["KD_unlearning"](
            unlearning_model, clients_dict[i], cfg
        )
        distiller_unlearning = torch.nn.DataParallel(distiller_unlearning.cuda())
        experiment_name = experiment_name + "_unlearning"
        unlearning_trainer = trainer_dict["unlearning"](
            experiment_name, distiller_unlearning, clients_train_dataloader[i], clients_train_dataloader[i], cfg
        )
        unlearning_trainer.train()
        clients_dict[i].load_state_dict(unlearning_trainer.get_student())

    cfg.defrost()
    cfg.SOLVER.LR = 0.001
    cfg.SOLVER.EPOCHS = 10
    cfg.SOLVER.BATCH_SIZE = 32
    cfg.freeze()
    show_cfg(cfg)

    clients_train_dataloader, server_train_dataloader, test_loader, server_data_num, num_classes \
        = get_fed_dataset_strong(cfg)
    for i in range(clients_num):
        clients_dataset_len[i] = len(clients_train_dataloader[i])

    for i in range(len(clients_dict)):
        clients_dict[i].cuda()
        test_model(clients_dict[i], test_loader)
    global_model.cuda()
    test_model(global_model, test_loader)

    print("-------------before unlearning agg-----------")
    print_time("staring unlearning agg")
    global_model = tinyimagenet_model_dict[cfg.FEDERATED.SERVER][0](num_classes=num_classes)
    fed_distiller = distiller_dict["fed_KD_ours"](
        global_model, clients_dict, cfg, clients_dataset_len
    )
    fed_distiller = torch.nn.DataParallel(fed_distiller.cuda())
    feb_trainer = trainer_dict["fed_ours"](
        experiment_name, fed_distiller, server_train_dataloader, test_loader, cfg
    )
    feb_trainer.train()
    global_model.load_state_dict(feb_trainer.get_student())
    print_time("ending unlearning agg")
    test_model(global_model, test_loader)

    cfg.defrost()
    cfg.SOLVER.EPOCHS = 10
    cfg.SOLVER.LR = 0.001
    cfg.EXPERIMENT.TAG = cfg.EXPERIMENT.TAG + '_unlearning'
    cfg.freeze()
    show_cfg(cfg)
    clients_num = cfg.FEDERATED.CLIENT_NUM - len(unlearning_clients)
    recover_clients_dict = {}
    recover_clients_datalodaer = {}
    recover_clients_datalen = {}
    exit_client_num = len(unlearning_clients)

    for i in range(len(clients_dict) - exit_client_num):
        recover_clients_dict[i] = clients_dict[i]
        recover_clients_datalodaer[i] = clients_train_dataloader[i]
        recover_clients_datalen[i] = len(clients_train_dataloader[i])

    distiller_server_to_client = distiller_dict[cfg.DISTILLER.TYPE](
        communication_model, global_model, cfg
    )

    distiller_server_to_client = torch.nn.DataParallel(distiller_server_to_client.cuda())
    fed_distiller = distiller_dict["fed_KD_ours"](
        global_model, recover_clients_dict, cfg, recover_clients_datalen
    )
    fed_distiller = torch.nn.DataParallel(fed_distiller.cuda())
    feb_trainer.set_distiller(fed_distiller)

    experiment_name = experiment_name + "_distribute"
    distribute_trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller_server_to_client, server_train_dataloader, test_loader, cfg
    )

    print("--------start recover---------")
    print_time("start recover")
    for r in range(recover_round):
        print(f"------------------recover round {r}------------------")
        distiller_server_to_client = distiller_dict[cfg.DISTILLER.TYPE](
            communication_model, global_model, cfg
        )
        distiller_server_to_client = torch.nn.DataParallel(distiller_server_to_client.cuda())
        distribute_trainer.set_distiller(distiller_server_to_client)
        distribute_trainer.train()
        communication_model.load_state_dict(distribute_trainer.get_student())

        for i in range(clients_num):
            recover_clients_dict[i].load_state_dict(communication_model.state_dict())

        for i in range(clients_num):
            print(f"------------client {i} is recover training----------")
            train_client(
                model=recover_clients_dict[i],
                train_dataloader=recover_clients_datalodaer[i],
                test_dataloader=test_loader,
                cfg=cfg
            )
            backdoor_test_model(recover_clients_dict[i], test_loader)
        for i in range(clients_num):
            recover_clients_dict[i].cuda()

        fed_distiller = distiller_dict["fed_KD_ours"](
            global_model, recover_clients_dict, cfg, recover_clients_datalen
        )
        fed_distiller = torch.nn.DataParallel(fed_distiller.cuda())
        feb_trainer.set_distiller(fed_distiller)
        feb_trainer.train()
        global_model.load_state_dict(feb_trainer.get_student())

        print("--------------global model test----------------")
        test_model(global_model, test_loader)
        backdoor_test_model(global_model, test_loader)
        save_ckpt(clients_model=recover_clients_dict, server_model=global_model,
                  communication_model=communication_model, cfg=cfg, epoch=r)
    print_time("end recover")
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
