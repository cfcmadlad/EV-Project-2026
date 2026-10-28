from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from src.config import Technique

N_CELLS = 4
SEEDS = (0, 1, 2)
VALIDATION_FRACTION = 0.15
MAX_EPOCHS = 400
PATIENCE = 40
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
CONFIGS = ((64, 1e-3), (128, 1e-3), (64, 3e-3), (128, 3e-3))
SOC_COLUMNS = slice(2, 2 + N_CELLS)
CAPACITY_COLUMNS = slice(2 + N_CELLS, 2 + 2 * N_CELLS)
R0_COLUMNS = slice(2 + 2 * N_CELLS, 2 + 3 * N_CELLS)
TAIL_COLUMNS = slice(2 + 3 * N_CELLS, None)
SPREAD_COLUMNS = slice(0, 2)


@dataclass(frozen=True, slots=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_split(technique: np.ndarray, seed: int, test_fraction: float) -> Split:
    index = np.arange(technique.size)
    train_all, test = train_test_split(
        index, test_size=test_fraction, random_state=seed, stratify=technique
    )
    train, validation = train_test_split(
        train_all,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        stratify=technique[train_all],
    )
    return Split(train, validation, test)


@dataclass(frozen=True, slots=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


def fit_scaler(values: np.ndarray) -> Scaler:
    return Scaler(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-12))


class Mlp(nn.Module):
    def __init__(self, n_features: int, width: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, n_out),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class CellGru(nn.Module):
    def __init__(self, n_features: int, width: int, n_out: int) -> None:
        super().__init__()
        hidden = width // 2
        self.gru = nn.GRU(3, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + n_features - 3 * N_CELLS, width),
            nn.ReLU(),
            nn.Linear(width, n_out),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens, tail = split_inputs(features)
        _, state = self.gru(tokens)
        return self.head(torch.cat((state[0], state[1], tail), dim=1))


class CellTransformer(nn.Module):
    def __init__(self, n_features: int, width: int, n_out: int) -> None:
        super().__init__()
        dim = width // 2
        self.embed = nn.Linear(3, dim)
        self.position = nn.Parameter(torch.zeros(1, N_CELLS, dim))
        layer = nn.TransformerEncoderLayer(
            dim, 2, dim_feedforward=2 * dim, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(dim + n_features - 3 * N_CELLS, width), nn.ReLU(), nn.Linear(width, n_out)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens, tail = split_inputs(features)
        encoded = self.encoder(self.embed(tokens) + self.position)
        return self.head(torch.cat((encoded.mean(dim=1), tail), dim=1))


def split_inputs(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.stack(
        (features[:, SOC_COLUMNS], features[:, CAPACITY_COLUMNS], features[:, R0_COLUMNS]), dim=2
    )
    tail = torch.cat((features[:, SPREAD_COLUMNS], features[:, TAIL_COLUMNS]), dim=1)
    return tokens, tail


MODELS = {"mlp": Mlp, "cell_gru": CellGru, "cell_transformer": CellTransformer}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)


@dataclass(frozen=True, slots=True)
class Fit:
    state: dict[str, torch.Tensor]
    validation_loss: float
    epochs: int
    seconds: float


def fit(
    kind: str,
    width: int,
    learning_rate: float,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> Fit:
    seed_everything(seed)
    model = MODELS[kind](x_train.shape[1], width, y_train.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    start = perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epochs = epoch
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator).to(device)
        for first in range(0, order.numel(), BATCH_SIZE):
            batch = order[first : first + BATCH_SIZE]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(nn.functional.mse_loss(model(x_val), y_val))
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    return Fit(best_state, best, epochs, perf_counter() - start)


def build(kind: str, n_features: int, n_out: int, width: int, state: dict) -> nn.Module:
    model = MODELS[kind](n_features, width, n_out)
    model.load_state_dict(state)
    return model.eval()


def predict(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model(features).numpy()


def median_time_per_pack(model: nn.Module, features: torch.Tensor, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        start = perf_counter()
        predict(model, features)
        times.append(perf_counter() - start)
    return float(np.median(times)) / features.shape[0]


def scores(y_true: np.ndarray, y_pred: np.ndarray, technique: np.ndarray) -> dict:
    groups = {"overall": np.ones_like(technique, dtype=bool)} | {
        t.key: technique == t for t in Technique
    }
    return {
        name: {
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        }
        for name, mask in groups.items()
    }


def summarise(per_seed: list[dict]) -> dict:
    out: dict = {}
    for target in per_seed[0]:
        out[target] = {}
        for group in per_seed[0][target]:
            out[target][group] = {}
            for metric in per_seed[0][target][group]:
                values = np.array([run[target][group][metric] for run in per_seed])
                out[target][group][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
    return out
