import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_squared_distance(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Computes fast Euclidean squared distance between x (B, D) and p (H, D)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    p2 = (p * p).sum(dim=-1).unsqueeze(0)
    return (x2 + p2 - 2.0 * (x @ p.t())).clamp_min(0.0)


class TRULayer(nn.Module):
    """
    Ternary Relational Unit (TRU) Layer.
    Maintains Positive (P), Negative (N), and Neutral/Context (U) reference prototypes.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        q = 1.0 / math.sqrt(in_features)
        self.P = nn.Parameter(torch.randn(out_features, in_features) * q)
        self.N = nn.Parameter(torch.randn(out_features, in_features) * q)
        self.U = nn.Parameter(torch.randn(out_features, in_features) * q)
        self.log_tau = nn.Parameter(torch.zeros(out_features))
        self.log_gamma = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-3
        gamma = F.softplus(self.log_gamma) + 1e-3

        dp = ((x[:, None, :] - self.P[None, :, :]) ** 2).mean(-1)
        dn = ((x[:, None, :] - self.N[None, :, :]) ** 2).mean(-1)
        du = ((x[:, None, :] - self.U[None, :, :]) ** 2).mean(-1)

        sp = torch.exp(-dp / tau)
        sn = torch.exp(-dn / tau)
        su = torch.exp(-du / tau)

        e = sp - sn
        g = torch.sigmoid(gamma * (su - 0.5 * (sp + sn)))
        return self.ln(torch.tanh(4.0 * e) * g + self.bias)


class TRN(nn.Module):
    """Two-layer Ternary Relational Network."""
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.a = TRULayer(in_features, hidden_dim)
        self.b = TRULayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.b(self.a(x)))


class DualLayer(nn.Module):
    """Dual-Prototype Baseline Layer (Positive and Negative Prototypes only)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        q = 1.0 / math.sqrt(in_features)
        self.P = nn.Parameter(torch.randn(out_features, in_features) * q)
        self.N = nn.Parameter(torch.randn(out_features, in_features) * q)
        self.log_tau = nn.Parameter(torch.zeros(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 1e-3
        dp = ((x[:, None, :] - self.P[None, :, :]) ** 2).mean(-1)
        dn = ((x[:, None, :] - self.N[None, :, :]) ** 2).mean(-1)
        e = torch.exp(-dp / tau) - torch.exp(-dn / tau)
        return self.ln(torch.tanh(4.0 * e) + self.bias)


class DualPrototype(nn.Module):
    """Two-layer Dual Prototype Network."""
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.a = DualLayer(in_features, hidden_dim)
        self.b = DualLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.b(self.a(x)))


class StandardMLP(nn.Module):
    """Standard MLP baseline with GELU activations."""
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelativeSoftmaxTRN(nn.Module):
    """Softmax-normalized relational TRN variant."""
    def __init__(self, d: int = 784, h: int = 256, c: int = 10):
        super().__init__()
        q = 1.0 / math.sqrt(d)
        self.p = nn.Parameter(torch.randn(h, d) * q)
        self.n = nn.Parameter(torch.randn(h, d) * q)
        self.u = nn.Parameter(torch.randn(h, d) * q)
        self.t = nn.Parameter(torch.zeros(h))
        self.g = nn.Parameter(torch.zeros(h))
        self.b = nn.Parameter(torch.zeros(h))
        self.ln = nn.LayerNorm(h)
        self.out = nn.Linear(h, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.t) + 1e-3
        gamma = F.softplus(self.g) + 1e-3
        dp = fast_squared_distance(x, self.p)
        dn = fast_squared_distance(x, self.n)
        du = fast_squared_distance(x, self.u)
        logits = torch.stack((-dp / tau, -dn / tau, -du / tau), dim=-1)
        w = F.softmax(logits, dim=-1)
        wp, wn, wu = w.unbind(dim=-1)
        evidence = wp - wn
        context = wu - 0.5 * (wp + wn)
        gate = torch.sigmoid(gamma * context)
        h = torch.tanh(2.0 * evidence) * gate + self.b
        return self.out(self.ln(h))


class ProjectedRelativeTRN(nn.Module):
    """Subspace-projected relational TRN variant."""
    def __init__(self, d: int = 784, k: int = 48, h: int = 256, c: int = 10):
        super().__init__()
        self.project = nn.Linear(d, k)
        q = 1.0 / math.sqrt(k)
        self.p = nn.Parameter(torch.randn(h, k) * q)
        self.n = nn.Parameter(torch.randn(h, k) * q)
        self.u = nn.Parameter(torch.randn(h, k) * q)
        self.t = nn.Parameter(torch.zeros(h))
        self.g = nn.Parameter(torch.zeros(h))
        self.kappa = nn.Parameter(torch.zeros(h))
        self.b = nn.Parameter(torch.zeros(h))
        self.ln = nn.LayerNorm(h)
        self.out = nn.Linear(h, c)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.project(x)
        tau = F.softplus(self.t) + 1e-3
        gamma = F.softplus(self.g) + 1e-3
        kappa = F.softplus(self.kappa)
        dp = fast_squared_distance(z, self.p)
        dn = fast_squared_distance(z, self.n)
        du = fast_squared_distance(z, self.u)
        logits = torch.stack((-dp / tau, -dn / tau, -du / tau), dim=-1)
        w = F.softmax(logits, dim=-1)
        wp, wn, wu = w.unbind(dim=-1)
        evidence = wp - wn
        context = wu - 0.5 * (wp + wn)
        z_norm = z.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        u_norm = self.u.norm(dim=-1).unsqueeze(0).clamp_min(self.eps)
        log_norm_gap = torch.log(z_norm) - torch.log(u_norm)
        norm_context = -log_norm_gap.abs()
        gate_logits = gamma * context + kappa * norm_context
        gate = torch.sigmoid(gate_logits)
        h = torch.tanh(2.0 * evidence) * gate + self.b
        return self.out(self.ln(h))
