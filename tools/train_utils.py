import os
import random
import time

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch import nn, optim
from tqdm import tqdm

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.engine.cfg import CFG as cfg


def log_file_path(experiment_name, cfg):
    struct_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d_%H:%M:%S", struct_time)
    file_path = "output_log/" + cfg.DATASET.TYPE + "/" + experiment_name + "_" + time_string + ".txt"
    return file_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_ckpt(clients_model, server_model, communication_model, cfg, epoch):
    file_path = "ckpt/" + cfg.DATASET.TYPE + "/" + cfg.EXPERIMENT.TAG
    epoch_dir = os.path.join(file_path, str(epoch))

    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Checkpoint directory for epoch {epoch} does not exist: {epoch_dir}")

    server_model_path = os.path.join(epoch_dir, "server_model.pth")
    if os.path.exists(server_model_path):
        server_model.load_state_dict(torch.load(server_model_path))
        print(f"Server model loaded from {server_model_path}")
    else:
        raise FileNotFoundError(f"Server model checkpoint not found at {server_model_path}")

    communication_model_path = os.path.join(epoch_dir, "communication_model.pth")
    if os.path.exists(communication_model_path):
        communication_model.load_state_dict(torch.load(communication_model_path))
        print(f"communication model loaded from {server_model_path}")
    else:
        raise FileNotFoundError(f"communication_model.pth checkpoint not found at {communication_model_path}")

    for client_id, client_model in clients_model.items():
        client_model_path = os.path.join(epoch_dir, f"client_{client_id}_model.pth")
        if os.path.exists(client_model_path):
            client_model.load_state_dict(torch.load(client_model_path))
            print(f"Client {client_id} model loaded from {client_model_path}")
        else:
            raise FileNotFoundError(f"Client {client_id} model checkpoint not found at {client_model_path}")

    return server_model, clients_model, communication_model


def load_backdoor_ckpt(clients_model, server_model, communication_model, cfg):
    file_path = "ckpt/" + cfg.DATASET.TYPE + "/" + cfg.EXPERIMENT.TAG
    epoch_dir = os.path.join(file_path, "backdoor")

    if not os.path.exists(epoch_dir):
        raise FileNotFoundError(f"Checkpoint directory for backdoor_old does not exist: {epoch_dir}")

    server_model_path = os.path.join(epoch_dir, "server_model.pth")
    if os.path.exists(server_model_path):
        server_model.load_state_dict(torch.load(server_model_path))
        print(f"Server model loaded from {server_model_path}")
    else:
        raise FileNotFoundError(f"Server model checkpoint not found at {server_model_path}")

    communication_model_path = os.path.join(epoch_dir, "communication_model.pth")
    if os.path.exists(communication_model_path):
        communication_model.load_state_dict(torch.load(communication_model_path))
        print(f"communication model loaded from {server_model_path}")
    else:
        raise FileNotFoundError(f"communication_model.pth checkpoint not found at {communication_model_path}")

    for client_id, client_model in clients_model.items():
        client_model_path = os.path.join(epoch_dir, f"client_{client_id}_model.pth")
        if os.path.exists(client_model_path):
            client_model.load_state_dict(torch.load(client_model_path))
            print(f"Client {client_id} model loaded from {client_model_path}")
        else:
            raise FileNotFoundError(f"Client {client_id} model checkpoint not found at {client_model_path}")

    return server_model, clients_model, communication_model


def save_backdoor_ckpt(clients_model, server_model, communication_model, cfg):
    file_path = "ckpt/" + cfg.DATASET.TYPE + "/" + cfg.EXPERIMENT.TAG + '/backdoor'
    os.makedirs(file_path, exist_ok=True)
    server_model_path = os.path.join(file_path, "server_model.pth")
    torch.save(server_model.state_dict(), server_model_path)
    communication_model_path = os.path.join(file_path, "communication_model.pth")
    torch.save(communication_model.state_dict(), communication_model_path)
    for client_id, client_model in clients_model.items():
        client_model_path = os.path.join(file_path, f"client_{client_id}_model.pth")
        torch.save(client_model.state_dict(), client_model_path)


def save_ckpt(clients_model, server_model, communication_model, cfg, epoch):
    file_path = "ckpt/" + cfg.DATASET.TYPE + "/" + cfg.EXPERIMENT.TAG
    os.makedirs(file_path, exist_ok=True)

    epoch_dir = os.path.join(file_path, str(epoch))
    os.makedirs(epoch_dir, exist_ok=True)

    server_model_path = os.path.join(epoch_dir, "server_model.pth")
    torch.save(server_model.state_dict(), server_model_path)

    communication_model_path = os.path.join(epoch_dir, "communication_model.pth")
    torch.save(communication_model.state_dict(), communication_model_path)

    for client_id, client_model in clients_model.items():
        client_model_path = os.path.join(epoch_dir, f"client_{client_id}_model.pth")
        torch.save(client_model.state_dict(), client_model_path)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(train_loader, desc='Training'):
        inputs, labels, index = data
        inputs = inputs[0]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total * 100
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[0], labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total * 100
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')


def train_client(model, train_dataloader, test_dataloader, cfg):
    lr = cfg.FEDERATED.CLIENT_LR
    num_epochs = cfg.FEDERATED.LOCAL_EPOCH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0:
            validate_one_epoch(model, test_dataloader, criterion, device)
        scheduler.step()
    model.to("cpu")


def test_model(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    validate_one_epoch(model, test_dataloader, criterion, device)


def backdoor_test_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    backdoor_success = 0
    backdoor_total = 0
    target_class = 1
    with torch.no_grad():
        for data in tqdm(val_loader, desc='back door Validation'):
            image, targets = data
            backdoor_inputs = add_trigger(image)
            backdoor_inputs = backdoor_inputs.to(device)
            labels = targets.to(device)

            backdoor_output = model(backdoor_inputs)
            _, backdoor_pred = torch.max(backdoor_output[0].data, 1)
            backdoor_total += labels.size(0)
            backdoor_success += (backdoor_pred == target_class).sum().item()
    print(f"backdoor attack acc:{100 * backdoor_success / backdoor_total:.3f}%")


def test_client_data(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(val_loader, desc='Validation'):
            image, targets, index = data
            inputs = image[0]
            inputs = inputs.to(device)
            labels = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[0], labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total * 100
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
