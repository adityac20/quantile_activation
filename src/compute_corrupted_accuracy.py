import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

from tqdm import tqdm

from src.models.resnet import ResNet18
from src.datasets.cifar10c import DistortedCIFAR10, list_distorions, list_severity


@torch.no_grad()
def compute_accuracy(model, dataset, device):
    model.train()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Update the statistics of the model using the dataset
    for _ in tqdm(range(20)):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)

    model.eval()
    correct = 0
    total = 0
    with tqdm(total=len(dataloader), leave=False) as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(accuracy=100 * correct / total)
            pbar.update(1)
    return 100 * correct / total


def main():
    model_path = "./data/run01/best_model_resnet18_quant_cifar10.pth"
    model = ResNet18(activation_type="quant", num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    file_results_path = "./results/resnet18_quant_cifar10_corrupted.csv"
    with open(file_results_path, "w") as f:
        f.write("distortion,severity,accuracy\n")

    root = "~/Documents/Datasets_Global/"
    root = os.path.expanduser(root)
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    for distortion in list_distorions:
        for severity in list_severity:
            dataset = DistortedCIFAR10(root, distortion, severity, transform)
            accuracy = compute_accuracy(model, dataset, device)
            with open(file_results_path, "a") as f:
                f.write(f"{distortion},{severity},{accuracy}\n")


if __name__ == "__main__":
    main()
