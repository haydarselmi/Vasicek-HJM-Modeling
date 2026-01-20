"""Autoencoder-based dimensionality reduction for yield curve data.

This module provides an alternative to PCA for nonlinear dimensionality
reduction of yield curve data. It implements a feedforward autoencoder
architecture using PyTorch for capturing complex patterns in interest
rate movements.

Optional module: Requires torch package (install with pip install torch
or pip install hjm-vasicek-modeling[autoencoder]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from src.analysis.pca import PCAResult, compute_pca
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.loader import InterestRateDataset

logger = get_logger("analysis.autoencoder")

# Try to import torch, provide clear error if not available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not installed. Install with: pip install torch "
        "or pip install hjm-vasicek-modeling[autoencoder]"
    )


def _check_torch_available() -> None:
    """Raise ImportError if torch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for autoencoder functionality. "
            "Install with: pip install torch or pip install hjm-vasicek-modeling[autoencoder]"
        )


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AutoencoderResult:
    """Results from autoencoder training and analysis.

    Attributes:
        model: The trained autoencoder model.
        latent_dim: Dimensionality of the latent space.
        train_loss_history: Training loss per epoch.
        val_loss_history: Validation loss per epoch (if validation data provided).
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss (if applicable).
        reconstruction_error: MSE on the training data after training.
        training_date: When training was performed.
    """

    model: "YieldCurveAutoencoder"
    latent_dim: int
    train_loss_history: list[float]
    val_loss_history: list[float] = field(default_factory=list)
    final_train_loss: float = 0.0
    final_val_loss: float | None = None
    reconstruction_error: float = 0.0
    training_date: datetime = field(default_factory=datetime.now)


@dataclass
class ComparisonResult:
    """Results from comparing PCA and autoencoder dimensionality reduction.

    Attributes:
        pca_reconstruction_error: Mean squared reconstruction error for PCA.
        autoencoder_reconstruction_error: Mean squared reconstruction error for autoencoder.
        pca_explained_variance: Total variance explained by PCA components.
        n_components: Number of components/latent dimensions used.
        pca_result: Full PCA analysis result.
        autoencoder: The trained autoencoder model.
        autoencoder_history: Training loss history.
    """

    pca_reconstruction_error: float
    autoencoder_reconstruction_error: float
    pca_explained_variance: float
    n_components: int
    pca_result: PCAResult
    autoencoder: "YieldCurveAutoencoder"
    autoencoder_history: dict[str, list[float]]


# =============================================================================
# Neural Network Model
# =============================================================================


if TORCH_AVAILABLE:

    class YieldCurveAutoencoder(nn.Module):
        """Feedforward autoencoder for yield curve dimensionality reduction.

        Architecture: N → 64 → 32 → K → 32 → 64 → N
        Where N is the input dimension (number of maturities) and K is the
        latent dimension.

        Activation: ReLU for hidden layers, linear for output.

        Attributes:
            input_dim: Number of maturity points (input features).
            latent_dim: Dimension of the latent (encoded) space.
            encoder: Encoder network (input → latent).
            decoder: Decoder network (latent → output).
        """

        def __init__(self, input_dim: int, latent_dim: int = 3) -> None:
            """Initialize the autoencoder.

            Args:
                input_dim: Number of input features (maturity points).
                latent_dim: Dimension of the latent space (default: 3).
            """
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder: N → 64 → 32 → K
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim),
            )

            # Decoder: K → 32 → 64 → N
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

            logger.debug(
                "Created autoencoder: %d → 64 → 32 → %d → 32 → 64 → %d",
                input_dim,
                latent_dim,
                input_dim,
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input to latent representation.

            Args:
                x: Input tensor of shape (batch_size, input_dim).

            Returns:
                Latent representation of shape (batch_size, latent_dim).
            """
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode latent representation to original space.

            Args:
                z: Latent tensor of shape (batch_size, latent_dim).

            Returns:
                Reconstructed tensor of shape (batch_size, input_dim).
            """
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: encode then decode.

            Args:
                x: Input tensor of shape (batch_size, input_dim).

            Returns:
                Reconstructed tensor of shape (batch_size, input_dim).
            """
            z = self.encode(x)
            return self.decode(z)

        def encode_numpy(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """Encode numpy array to latent space.

            Args:
                data: Input array of shape (n_samples, input_dim).

            Returns:
                Encoded array of shape (n_samples, latent_dim).
            """
            self.eval()
            with torch.no_grad():
                x = torch.from_numpy(data.astype(np.float32))
                z = self.encode(x)
                return z.numpy()

        def decode_numpy(self, latent: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """Decode latent vectors to original space.

            Args:
                latent: Latent array of shape (n_samples, latent_dim).

            Returns:
                Decoded array of shape (n_samples, input_dim).
            """
            self.eval()
            with torch.no_grad():
                z = torch.from_numpy(latent.astype(np.float32))
                x = self.decode(z)
                return x.numpy()

        def reconstruct_numpy(
            self, data: npt.NDArray[np.float32]
        ) -> npt.NDArray[np.float32]:
            """Reconstruct data through encode-decode cycle.

            Args:
                data: Input array of shape (n_samples, input_dim).

            Returns:
                Reconstructed array of shape (n_samples, input_dim).
            """
            self.eval()
            with torch.no_grad():
                x = torch.from_numpy(data.astype(np.float32))
                reconstructed = self.forward(x)
                return reconstructed.numpy()

else:
    # Placeholder class when torch is not available
    class YieldCurveAutoencoder:  # type: ignore[no-redef]
        """Placeholder for YieldCurveAutoencoder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Raise error on instantiation."""
            _check_torch_available()


# =============================================================================
# Training Functions
# =============================================================================


def train_autoencoder(
    model: "YieldCurveAutoencoder",
    data: npt.NDArray[np.float32],
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_data: npt.NDArray[np.float32] | None = None,
) -> dict[str, list[float]]:
    """Train the autoencoder on yield curve data.

    Uses MSE loss and Adam optimizer for training. Optionally tracks
    validation loss if validation data is provided.

    Args:
        model: The autoencoder model to train.
        data: Training data of shape (n_samples, n_features).
        n_epochs: Number of training epochs (default: 100).
        batch_size: Batch size for training (default: 32).
        learning_rate: Learning rate for Adam optimizer (default: 0.001).
        validation_data: Optional validation data for tracking val loss.

    Returns:
        Dictionary with training history:
        - 'train_loss': List of training loss per epoch
        - 'val_loss': List of validation loss per epoch (if val data provided)
    """
    _check_torch_available()

    logger.info(
        "Training autoencoder: %d samples, %d epochs, batch_size=%d, lr=%.4f",
        len(data),
        n_epochs,
        batch_size,
        learning_rate,
    )

    # Create data loaders
    train_tensor = torch.from_numpy(data.astype(np.float32))
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if validation_data is not None:
        val_tensor = torch.from_numpy(validation_data.astype(np.float32))
        val_dataset = TensorDataset(val_tensor, val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # Validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch_x, _ in val_loader:
                    reconstructed = model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    val_loss += loss.item()
                    n_val_batches += 1

            avg_val_loss = val_loss / n_val_batches
            history["val_loss"].append(avg_val_loss)
            model.train()

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if val_loader is not None:
                logger.info(
                    "Epoch %d/%d - train_loss: %.6f, val_loss: %.6f",
                    epoch + 1,
                    n_epochs,
                    avg_train_loss,
                    history["val_loss"][-1],
                )
            else:
                logger.info(
                    "Epoch %d/%d - train_loss: %.6f",
                    epoch + 1,
                    n_epochs,
                    avg_train_loss,
                )

    logger.info("Training complete. Final train_loss: %.6f", history["train_loss"][-1])

    return history


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_dimensionality_methods(
    dataset: "InterestRateDataset",
    n_components: int = 3,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_changes: bool = True,
) -> ComparisonResult:
    """Compare PCA and autoencoder for dimensionality reduction.

    Trains an autoencoder and computes PCA with the same number of
    components/latent dimensions, then compares reconstruction errors.

    Args:
        dataset: Interest rate dataset to analyze.
        n_components: Number of PCA components / autoencoder latent dim.
        n_epochs: Training epochs for autoencoder.
        batch_size: Batch size for autoencoder training.
        learning_rate: Learning rate for autoencoder.
        use_changes: If True, analyze rate changes; if False, rate levels.

    Returns:
        ComparisonResult with reconstruction errors and trained models.
    """
    _check_torch_available()

    logger.info(
        "Comparing PCA vs Autoencoder with %d components/latent dimensions",
        n_components,
    )

    # Get data matrix
    if use_changes:
        from src.analysis.pca import compute_yield_curve_changes, _get_changes_matrix

        changes = compute_yield_curve_changes(dataset)
        data = _get_changes_matrix(changes)
    else:
        data = dataset.rates.copy()

    # Standardize data for autoencoder
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0) + 1e-8
    data_normalized = ((data - data_mean) / data_std).astype(np.float32)

    # --- PCA ---
    pca_result = compute_pca(dataset, n_components=n_components, use_changes=use_changes)

    # Compute PCA reconstruction error
    from src.analysis.pca import reconstruct_curve

    pca_reconstructed = np.zeros_like(data)
    for i, changes_row in enumerate(data - data_mean):
        pca_reconstructed[i] = reconstruct_curve(pca_result, n_components, changes_row) + data_mean
    pca_mse = np.mean((data - pca_reconstructed) ** 2)

    logger.info("PCA reconstruction MSE: %.6f", pca_mse)

    # --- Autoencoder ---
    input_dim = data.shape[1]
    model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=n_components)

    history = train_autoencoder(
        model=model,
        data=data_normalized,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Compute autoencoder reconstruction error (on original scale)
    ae_reconstructed_normalized = model.reconstruct_numpy(data_normalized)
    ae_reconstructed = ae_reconstructed_normalized * data_std + data_mean
    ae_mse = np.mean((data - ae_reconstructed) ** 2)

    logger.info("Autoencoder reconstruction MSE: %.6f", ae_mse)

    # Total explained variance for PCA
    total_pca_variance = float(pca_result.cumulative_variance[-1]) if len(pca_result.cumulative_variance) > 0 else 0.0

    return ComparisonResult(
        pca_reconstruction_error=float(pca_mse),
        autoencoder_reconstruction_error=float(ae_mse),
        pca_explained_variance=total_pca_variance,
        n_components=n_components,
        pca_result=pca_result,
        autoencoder=model,
        autoencoder_history=history,
    )


def get_latent_encodings(
    model: "YieldCurveAutoencoder",
    data: npt.NDArray[np.float64],
    normalize: bool = True,
) -> npt.NDArray[np.float32]:
    """Get latent space encodings for data.

    Args:
        model: Trained autoencoder model.
        data: Data to encode of shape (n_samples, n_features).
        normalize: If True, standardize data before encoding.

    Returns:
        Latent encodings of shape (n_samples, latent_dim).
    """
    _check_torch_available()

    data_float = data.astype(np.float32)

    if normalize:
        data_mean = np.mean(data_float, axis=0)
        data_std = np.std(data_float, axis=0) + 1e-8
        data_float = (data_float - data_mean) / data_std

    return model.encode_numpy(data_float)
