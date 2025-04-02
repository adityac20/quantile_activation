import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from src.models.resnet import ResNet18

from src.datasets.cifar10 import get_cifar10
from src.datasets.cifar100 import get_cifar100


def get_dataset(dataset_name: str):
    if dataset_name == "cifar10":
        return get_cifar10(root="~/Documents/Datasets_Global/")
    elif dataset_name == "cifar100":
        return get_cifar100(root="~/Documents/Datasets_Global/")


def get_model(model_name: str, activation_type: str, dataset_name: str):
    if model_name == "resnet18" and dataset_name == "cifar10":
        return ResNet18(activation_type=activation_type, num_classes=10)
    elif model_name == "resnet18" and dataset_name == "cifar100":
        return ResNet18(activation_type=activation_type, num_classes=100)


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    loss_total = 0
    total = 0
    with tqdm(total=len(trainloader), leave=False) as pbar:
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, loss_watershed = model(inputs)
            loss = criterion(outputs, labels) + loss_watershed * (1.0 / len(trainloader))
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * inputs.size(0)
            total += inputs.size(0)
            pbar.set_postfix(loss=loss_total / total)
            pbar.update(1)
    return loss_total / total


def test_one_epoch(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    correct = 0
    total = 0
    with tqdm(total=len(testloader), leave=False) as pbar:
        for i, (inputs, labels) in enumerate(testloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=correct / total)
            pbar.update(1)
    return correct / total


def train(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    params: dict,
):
    best_acc = 0
    run_id = 5
    path_prefix = f"./data/run{run_id:02d}/"
    os.makedirs(path_prefix, exist_ok=True)
    path_postfix = f"{params['model_name']}_{params['activation_type']}_{params['dataset']}"
    path_best_model = path_prefix + f"best_model_{path_postfix}.pth"
    log_file = path_prefix + f"log_{path_postfix}.txt"

    with open(log_file, "w") as f:
        f.write(f"Epoch, Train Loss, Test Acc\n")
    with tqdm(total=params["num_epochs"], leave=False) as pbar_whole:
        for epoch in range(params["num_epochs"]):
            train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            test_acc = test_one_epoch(model, testloader, criterion, device)

            if scheduler is not None:
                scheduler.step()

            pbar_whole.set_postfix(train_loss=train_loss, test_acc=test_acc)
            pbar_whole.update(1)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), path_best_model)

            with open(log_file, "a") as f:
                f.write(f"{epoch:03d}, {train_loss:.4f}, {test_acc:.4f}\n")


def main():
    params = {
        "num_epochs": 200,
        "batch_size": 128,
        "learning_rate": 0.1,
        "weight_decay": 0.0001,
        "dataset": "cifar100",
        "model_name": "resnet18",
        "activation_type": "quant",
        "flag_intermediate_quant": True,
    }
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(params["dataset"])
    trainset, testset, transform_train, transform_test = dataset

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = get_model(params["model_name"], params["activation_type"], params["dataset"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=params["learning_rate"],
        momentum=0.9,
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["num_epochs"])

    train(model, trainloader, testloader, criterion, optimizer, scheduler, device, params)


if __name__ == "__main__":
    main()
