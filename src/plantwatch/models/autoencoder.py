"""LSTM autoencoder — the deep challenger.

Trained only on *normal* sequences to reconstruct the input window; the per-window
reconstruction error is the anomaly score. This is the canonical deep approach for
multivariate time series.

PyTorch is an optional dependency. Install it with:

    pip install -e ".[deep]"

The rest of Phase 2 (baseline, PCA, evaluation) runs without it.
"""

from __future__ import annotations

import numpy as np

from plantwatch.models.base import AnomalyDetector

try:  # torch is optional
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _HAS_TORCH = False


if _HAS_TORCH:

    class _Seq2SeqAE(nn.Module):
        def __init__(self, n_features: int, hidden: int, latent: int) -> None:
            super().__init__()
            self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.output = nn.Linear(hidden, n_features)

        def forward(self, x):  # x: (B, T, F)
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])  # (B, latent)
            seeded = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            decoded, _ = self.decoder(seeded)
            return self.output(decoded)


class LSTMAutoencoder(AnomalyDetector):
    name = "lstm_autoencoder"

    def __init__(
        self,
        *,
        hidden: int = 32,
        latent: int = 16,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch not installed. Run: pip install -e '.[deep]'")
        self.hidden = hidden
        self.latent = latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self._model: _Seq2SeqAE | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def fit(self, X) -> LSTMAutoencoder:
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std = X.std(axis=(0, 1), keepdims=True) + 1e-6
        data = torch.from_numpy(self._standardize(X))

        self._model = _Seq2SeqAE(X.shape[2], self.hidden, self.latent)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(data))
            for start in range(0, len(data), self.batch_size):
                batch = data[perm[start : start + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self._model(batch), batch)
                loss.backward()
                opt.step()
        return self

    def score(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before score()")
        X = np.asarray(X, dtype=np.float32)
        data = torch.from_numpy(self._standardize(X))
        self._model.eval()
        with torch.no_grad():
            recon = self._model(data)
            err = ((recon - data) ** 2).mean(dim=(1, 2))
        return err.cpu().numpy()
