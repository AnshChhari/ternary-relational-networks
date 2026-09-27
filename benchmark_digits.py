import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from trn import StandardMLP, DualPrototype, TRN

OUT = Path('results')
OUT.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_data(seed: int = 42, noise: float = 0.0):
    d = load_digits()
    X = d.data.astype('float32') / 16.0
    y = d.target.astype('int64')
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xte = X[tr], X[te]
    if noise > 0.0:
        rng = np.random.default_rng(123)
        Xte = np.clip(Xte + rng.normal(0, noise, Xte.shape).astype('float32'), 0.0, 1.0)
    return Xtr, y[tr], Xte, y[te]


def fit(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 25, seed: int = 0):
    seed_all(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xb = torch.tensor(X)
    yb = torch.tensor(y)
    best_acc = 0.0
    best_state = None

    for _ in range(epochs):
        model.train()
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(xb).argmax(1) == yb).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)


def evaluate(model: torch.nn.Module, X: np.ndarray, y: np.ndarray):
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(torch.tensor(X)), dim=1).numpy()
    acc = float((p.argmax(1) == y).mean())
    loss = float(log_loss(y, p, labels=list(range(10))))
    return acc, loss


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    rows = []
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        Xtr, ytr, Xte, yte = make_data(seed=42, noise=0.0)
        Xte_noisy = make_data(seed=42, noise=0.20)[2]

        models = [
            ('MLP', lambda: StandardMLP(64, 64, 10)),
            ('DualPrototype', lambda: DualPrototype(64, 64, 10)),
            ('TRN', lambda: TRN(64, 64, 10))
        ]

        for name, ctor in models:
            seed_all(seed)
            m = ctor()
            t0 = time.time()
            fit(m, Xtr, ytr, epochs=25, seed=seed)
            sec = time.time() - t0

            acc, ll = evaluate(m, Xte, yte)
            accn, lln = evaluate(m, Xte_noisy, yte)

            record = {
                'seed': seed,
                'model': name,
                'clean_acc': acc,
                'noisy_acc': accn,
                'clean_logloss': ll,
                'noisy_logloss': lln,
                'params': count_parameters(m),
                'seconds': sec
            }
            rows.append(record)
            print(json.dumps(record))

    with open(OUT / 'benchmark_digits_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
