import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.model_selection import train_test_split

from trn import StandardMLP, DualPrototype, TRN, RelativeSoftmaxTRN, ProjectedRelativeTRN

ROOT = Path('trn_v2_results')
ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mnist(root='data', seed=0, noise_sigma=0.0, validation_fraction=0.15):
    transform = transforms.ToTensor()
    train_ds = MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = MNIST(root=root, train=False, download=True, transform=transform)

    x_train = train_ds.data.float().div(255.0).view(-1, 784).numpy().astype('float32')
    y_train = train_ds.targets.numpy().astype('int64')
    x_test = test_ds.data.float().div(255.0).view(-1, 784).numpy().astype('float32')
    y_test = test_ds.targets.numpy().astype('int64')

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=validation_fraction, stratify=y_train, random_state=seed
    )

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    if noise_sigma > 0.0:
        rng = np.random.default_rng(100000 + seed)
        x_test = np.clip(x_test + rng.normal(0.0, noise_sigma, x_test.shape).astype('float32'), 0.0, 1.0)

    return x_train, y_train, x_val, y_val, x_test, y_test


@torch.no_grad()
def evaluate(model, x, y, batch_size=2048):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(DEVICE)
        yb = torch.from_numpy(y[start:start + batch_size]).to(DEVICE)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction='sum').item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(yb)
    return correct / total, loss_sum / total


def train(model, x_train, y_train, x_val, y_val, epochs=25, batch_size=1024, seed=0):
    seed_all(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )

    best_acc = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate(model, x_val, y_val)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc


def main():
    seeds = [0, 1, 2]
    noise_levels = [0.0, 0.20]
    configs = [
        ('MLP-256', lambda: StandardMLP(784, 256, 10)),
        ('DualPrototype-256', lambda: DualPrototype(784, 256, 10)),
        ('OriginalTRN-256', lambda: TRN(784, 256, 10)),
        ('RelativeSoftmaxTRN-256', lambda: RelativeSoftmaxTRN(784, 256, 10)),
        ('ProjectedRelativeTRN-48-256', lambda: ProjectedRelativeTRN(784, 48, 256, 10)),
    ]

    all_rows = []
    for name, factory in configs:
        for noise in noise_levels:
            for seed in seeds:
                x_train, y_train, x_val, y_val, x_test, y_test = make_mnist(seed=seed, noise_sigma=noise)
                seed_all(seed)
                m = factory()
                t0 = time.time()
                m, val_acc = train(m, x_train, y_train, x_val, y_val, epochs=25, seed=seed)
                test_acc, test_loss = evaluate(m, x_test, y_test)
                all_rows.append({
                    'model': name, 'seed': seed, 'noise_sigma': noise,
                    'val_accuracy': val_acc, 'test_accuracy': test_acc,
                    'test_logloss': test_loss, 'seconds': time.time() - t0
                })
                print(json.dumps(all_rows[-1]))

    with open(ROOT / 'benchmark_mnist_rows.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == '__main__':
    main()
