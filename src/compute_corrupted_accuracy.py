import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

from tqdm import tqdm

from src.models.resnet import ResNet18
from src.datasets.cifar10c import DistortedCIFAR10, list_distorions, list_severity
from src.datasets.cifar100c import DistortedCIFAR100


def get_dataset(dataset_name, distortion, severity, transform):
    root = "~/Documents/Datasets_Global/"
    root = os.path.expanduser(root)
    if dataset_name == "cifar100":
        dataset = DistortedCIFAR100(root, distortion, severity, transform)
    elif dataset_name == "cifar10":
        dataset = DistortedCIFAR10(root, distortion, severity, transform)
    return dataset


def get_model(model_name, activation_type, dataset_name):
    if model_name == "resnet18" and dataset_name == "cifar100":
        model = ResNet18(activation_type=activation_type, num_classes=100)
    elif model_name == "resnet18" and dataset_name == "cifar10":
        model = ResNet18(activation_type=activation_type, num_classes=10)
    return model


@torch.no_grad()
def compute_accuracy(model, dataset_train, dataset_test, device):
    model.train()
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=4)
    # # Update the statistics of the model using the dataset
    for _ in tqdm(range(2)):
        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)

    model.eval()
    correct = 0
    total = 0
    with tqdm(total=len(dataloader_test), leave=False) as pbar:
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(accuracy=100 * correct / total)
            pbar.update(1)
    return 100 * correct / total


def main():
    params = {
        "dataset": "cifar100",  # cifar100, cifar10
        "model_name": "resnet18",  # resnet18
        "activation_type": "quant",  # quant, relu
    }
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_path = f"./data/run01/best_model_{params['model_name']}_{params['activation_type']}_{params['dataset']}.pth"
    model = get_model(params["model_name"], params["activation_type"], params["dataset"])
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    file_results_path = f"./results/{params['model_name']}_{params['activation_type']}_{params['dataset']}_corrupted_v3.csv"
    with open(file_results_path, "w") as f:
        f.write("distortion,severity,accuracy\n")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    for distortion in list_distorions:
        for severity in list_severity:
            dataset_train = get_dataset(params["dataset"], distortion, severity, transform_train)
            dataset_test = get_dataset(params["dataset"], distortion, severity, transform_test)
            accuracy = compute_accuracy(model, dataset_train, dataset_test, device)
            with open(file_results_path, "a") as f:
                f.write(f"{distortion},{severity},{accuracy}\n")


if __name__ == "__main__":
    main()
