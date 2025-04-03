# Quantile Activation

A PyTorch implementation of Quantile Activation, a novel activation function for neural networks with robustness to distribution shifts.

## Overview

Quantile Activation transforms the features of neural networks by mapping them through their empirical distribution function. This approach offers several advantages over traditional activation functions:

## Main Idea

The Main idea is to use **context distribution** associated with every neuron, and use quantile w.r.t the context distribution as the activation. 


## Installation

Easiest way to use the code is to copy the contents of the files `quantile_activation_accum.py` or `quantile_activation_base.py` into your project, and import the necessary functions.

Note that the underlying assumption of quantile activation is that context distribution reflects the distribution as suitable for inference. To ensure this,
1. If using `quantile_activation_accum.py`, make sure have `model.train()` and update the statistics using dummy feed-forward passes.
2. If using `quantile_activation_base.py`, make sure have large batch sizes

### Using Quantile Activation in Your Models

To use quantile activation in your own models:

```python
from src.quantile_activation_accum import quantile_activation_1d, quantile_activation_2d

# For 1D activations (e.g., fully connected layers)
self.act1d = quantile_activation_1d(n_features=512)

# For 2D activations (e.g., convolutional layers)
self.act2d = quantile_activation_2d(n_features=64)

# In forward pass
x = self.act1d(x)  # or self.act2d(x)
```

## Project Structure

```
├── data/                # Stores trained model data and results
├── notebooks/           # Jupyter notebooks for experiments
├── results/             # results on quant+cifar100/cifar10
├── src/                 # Source code
│   ├── datasets/        # Dataset loaders
│   ├── models/          # Neural network architectures
│   ├── __init__.py
│   ├── compute_corrupted_accuracy.py  # Evaluate on corrupted data
│   ├── quantile_activation_accum.py   # Quantile activation implementation
│   ├── quantile_activation_base.py    # Base quantile activation classes
│   ├── train_model.py                 # Main training script
└── LICENSE
```

## Generating results on CIFAR-10/CIFAR-10C

### Training a Model

To train a model with quantile activation:

```bash
python src/train_model.py
```

You can customize the training parameters by modifying the `params` dictionary in the `main()` function of `train_model.py`:

```python
params = {
    "num_epochs": 200,
    "batch_size": 128,
    "learning_rate": 0.1,
    "weight_decay": 0.0001,
    "dataset": "cifar100",      # "cifar10" or "cifar100"
    "model_name": "resnet18",   # Model architecture
    "activation_type": "quant", # "quant", "relu", "prelu", or "selu"
    "flag_intermediate_quant": True,
}
```

### Evaluating Robustness

To evaluate model robustness on corrupted data:

```bash
python src/compute_corrupted_accuracy.py
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Citation

If you use this code in your research, please cite:

```
@article{quant_act_aditya,
  author       = {Aditya Challa and
                  Sravan Danda and
                  Laurent Najman and
                  Snehanshu Saha},
  title        = {Quantile Activation: departing from single point estimation for better generalization across distortions},
  journal      = {arXiv:2405.11573},
  year         = {2024},
}
```