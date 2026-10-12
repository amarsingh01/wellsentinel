"""Phase 7: the sequence model itself.

A small masked GRU. It reads a stack of consecutive timesteps and predicts whether an
undesirable event will start inside the horizon. Two details matter more than the size of
the network.

* *Masking.* Real wells are missing about 9 to 11 of their 16 analogue sensors, and which
  ones differ per well. We fill the gaps with zero after normalisation and pass a parallel
  mask channel, so the model can tell "value is zero" from "value is absent".
* *Calibration over sharpness.* The business cares about false alarms per day and lead
  time, not about a pretty F1. We therefore report a threshold-free ranking score (ROC AUC,
  average precision) next to the 0.5-threshold numbers, and we never tune on the test fold.

Everything here is written for this project. It leans on ``torch`` for tensors and
autograd the same way the earlier phases leaned on ``lightgbm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


class MaskedGRU(nn.Module):
    """A GRU that is told which inputs were actually measured.

    The mask is concatenated to the input channels, so the cell sees both the value and
    its presence flag at every step. A tiny head turns the final hidden state into one
    logit: the risk that an event begins within the horizon.
    """

    def __init__(self, n_channels: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_channels * 2)
        self.gru = nn.GRU(
            input_size=n_channels * 2,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([x, mask], dim=-1)
        joined = self.input_norm(joined)
        output, _ = self.gru(joined)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class TrainConfig:
    """Knobs for one training run. Defaults are the small, honest ones."""

    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    seed: int = 42
    patience: int = 3
    device: str = "cpu"


@dataclass
class TrainResult:
    model: MaskedGRU | None = None
    best_epoch: int = -1
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.model is not None


def _to_tensor(x: np.ndarray, mask: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.float32, device=device),
    )


def train_sequence_model(
    x_train: np.ndarray,
    m_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    m_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train the masked GRU, early stopping on validation average precision.

    Class imbalance is handled by weighting the positive class in the loss, so the model
    is not rewarded for answering "no event" every time.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    result = TrainResult()
    if len(y_train) == 0 or len(y_val) == 0:
        return result
    if len(np.unique(y_train)) < 2:
        return result

    n_channels = x_train.shape[2]
    model = MaskedGRU(n_channels, hidden=config.hidden, layers=config.layers, dropout=config.dropout)
    model.to(config.device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    xb_tr, mb_tr = _to_tensor(x_train, m_train, config.device)
    yb_tr = torch.as_tensor(y_train, dtype=torch.float32, device=config.device)
    xb_va, mb_va = _to_tensor(x_val, m_val, config.device)
    yb_va = torch.as_tensor(y_val, dtype=torch.float32, device=config.device)

    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    n = len(y_train)

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n, device=config.device)
        epoch_loss = 0.0
        for start in range(0, n, config.batch_size):
            index = order[start : start + config.batch_size]
            optimiser.zero_grad()
            logits = model(xb_tr[index], mb_tr[index])
            loss = criterion(logits, yb_tr[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(index)

        model.eval()
        with torch.no_grad():
            val_logits = model(xb_va, mb_va)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        score = _average_precision(y_val, val_prob)
        result.history.append({"epoch": epoch, "loss": epoch_loss / n, "val_ap": score})

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        result.model = model
    return result


def predict(model: MaskedGRU, x: np.ndarray, mask: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Probability of an event within the horizon, per sequence."""
    model.eval()
    xb, mb = _to_tensor(x, mask, device)
    with torch.no_grad():
        return torch.sigmoid(model(xb, mb)).cpu().numpy()


def _average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Average precision without importing sklearn, so this module stands alone."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-probability)
    ranked = y_true[order]
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, len(ranked) + 1)
    recall = tp / y_true.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def score(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Ranking and thresholded metrics, computed here to keep the module self-contained."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability)
    out: dict[str, float] = {"n": float(len(y_true)), "positives": float(y_true.sum())}
    if len(np.unique(y_true)) < 2:
        return out

    predicted = (probability >= 0.5).astype(int)
    tp = float(((predicted == 1) & (y_true == 1)).sum())
    fp = float(((predicted == 1) & (y_true == 0)).sum())
    fn = float(((predicted == 0) & (y_true == 1)).sum())
    out["precision"] = round(tp / (tp + fp), 4) if tp + fp else 0.0
    out["recall"] = round(tp / (tp + fn), 4) if tp + fn else 0.0
    if out["precision"] + out["recall"]:
        out["f1"] = round(2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]), 4)
    else:
        out["f1"] = 0.0

    # ROC AUC via rank statistic (Mann-Whitney U), no sklearn dependency.
    pos = probability[y_true == 1]
    neg = probability[y_true == 0]
    if len(pos) and len(neg):
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order), dtype="float64")
        ranks[order] = np.arange(1, len(order) + 1)
        rank_pos = ranks[: len(pos)]
        auc = (rank_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        out["roc_auc"] = round(float(auc), 4)
    out["average_precision"] = round(_average_precision(y_true, probability), 4)
    out["base_rate"] = round(float(y_true.mean()), 4)
    return out


def lead_time_seconds(
    probabilities: np.ndarray,
    seconds_to_onset: np.ndarray,
    instance: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Mean seconds between the first alert in an instance and its next real onset.

    Negative means the alert fired after the onset. Only instances that contain both an
    alert and an onset contribute, and an instance contributes once.
    """
    leads: list[float] = []
    for name in np.unique(instance):
        mask = instance == name
        gaps = seconds_to_onset[mask].astype("float64")
        probs = probabilities[mask]
        # NaN gaps mean "no onset follows"; sort them last without relying on argsort's
        # na_position, which numpy does not support.
        order = np.argsort(np.where(np.isnan(gaps), np.inf, gaps))
        probs = probs[order]
        gaps = gaps[order]
        onset_rows = np.where(~np.isnan(gaps))[0]
        if not len(onset_rows):
            continue
        first_onset_position = onset_rows[0]
        alert_rows = np.where(probs >= threshold)[0]
        alert_rows = alert_rows[alert_rows < first_onset_position]
        if len(alert_rows):
            leads.append(float(gaps[first_onset_position]))
    return float(np.mean(leads)) if leads else float("nan")
