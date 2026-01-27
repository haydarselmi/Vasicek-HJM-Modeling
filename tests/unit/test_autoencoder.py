"""Unit tests for the autoencoder module.

This module tests the YieldCurveAutoencoder class and related training functions,
verifying encoder/decoder shapes, loss computation, and training behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if torch is not installed
torch = pytest.importorskip("torch")


class TestYieldCurveAutoencoder:
    """Tests for the YieldCurveAutoencoder neural network class."""

    def test_encoder_output_shape(self) -> None:
        """Test that encoder produces correct latent dimension output."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7  # Standard treasury maturities
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Create sample input
        batch_size = 32
        x = torch.randn(batch_size, input_dim)

        # Get encoded output
        encoded = model.encode(x)

        assert encoded.shape == (batch_size, latent_dim), (
            f"Expected encoded shape ({batch_size}, {latent_dim}), "
            f"got {encoded.shape}"
        )

    def test_decoder_output_shape(self) -> None:
        """Test that decoder reconstructs original dimension."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Create sample latent vector
        batch_size = 32
        z = torch.randn(batch_size, latent_dim)

        # Get decoded output
        decoded = model.decode(z)

        assert decoded.shape == (batch_size, input_dim), (
            f"Expected decoded shape ({batch_size}, {input_dim}), "
            f"got {decoded.shape}"
        )

    def test_forward_pass_shape(self) -> None:
        """Test that forward pass preserves input dimension."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        batch_size = 32
        x = torch.randn(batch_size, input_dim)

        # Full forward pass (encode + decode)
        reconstructed = model(x)

        assert reconstructed.shape == x.shape, (
            f"Expected output shape {x.shape}, got {reconstructed.shape}"
        )

    def test_different_latent_dimensions(self) -> None:
        """Test autoencoder with various latent dimension sizes."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        batch_size = 16
        x = torch.randn(batch_size, input_dim)

        for latent_dim in [1, 2, 3, 5]:
            model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
            encoded = model.encode(x)
            reconstructed = model(x)

            assert encoded.shape == (batch_size, latent_dim)
            assert reconstructed.shape == (batch_size, input_dim)

    def test_architecture_hidden_layers(self) -> None:
        """Test that model has expected hidden layer structure."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Check encoder has layers: input_dim -> 64 -> 32 -> latent_dim
        encoder_params = list(model.encoder.parameters())
        assert len(encoder_params) >= 6  # At least 3 layers with weights and biases

        # Check decoder has layers: latent_dim -> 32 -> 64 -> input_dim
        decoder_params = list(model.decoder.parameters())
        assert len(decoder_params) >= 6

    def test_model_is_trainable(self) -> None:
        """Test that model parameters have gradients after forward pass."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(32, input_dim)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, x)
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Gradients should be computed"


class TestLossComputation:
    """Tests for autoencoder loss computation."""

    def test_mse_loss_is_non_negative(self) -> None:
        """Test that MSE reconstruction loss is always non-negative."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(32, input_dim)
        reconstructed = model(x)
        loss = torch.nn.functional.mse_loss(reconstructed, x)

        assert loss.item() >= 0, "MSE loss should be non-negative"

    def test_perfect_reconstruction_has_zero_loss(self) -> None:
        """Test that identical input and output gives zero loss."""
        x = torch.randn(32, 7)
        loss = torch.nn.functional.mse_loss(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-7), (
            "Perfect reconstruction should have zero loss"
        )

    def test_loss_decreases_with_identity_init(self) -> None:
        """Test that a model can reduce loss with training."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 5
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Fixed input for reproducibility
        torch.manual_seed(42)
        x = torch.randn(100, input_dim)

        # Initial loss
        with torch.no_grad():
            initial_output = model(x)
            initial_loss = torch.nn.functional.mse_loss(initial_output, x)

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, x)
            loss.backward()
            optimizer.step()

        # Final loss
        with torch.no_grad():
            final_output = model(x)
            final_loss = torch.nn.functional.mse_loss(final_output, x)

        assert final_loss.item() < initial_loss.item(), (
            f"Loss should decrease with training: {initial_loss.item()} -> {final_loss.item()}"
        )


class TestTrainAutoencoder:
    """Tests for the train_autoencoder function."""

    def test_train_autoencoder_returns_history(self) -> None:
        """Test that training returns loss history."""
        from src.analysis.autoencoder import YieldCurveAutoencoder, train_autoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Create sample data
        np.random.seed(42)
        data = np.random.randn(100, input_dim).astype(np.float32)

        # Train for a few epochs
        history = train_autoencoder(
            model=model,
            data=data,
            n_epochs=5,
            batch_size=32,
            learning_rate=0.01,
        )

        assert "train_loss" in history, "History should contain train_loss"
        assert len(history["train_loss"]) == 5, "Should have loss for each epoch"

    def test_train_autoencoder_loss_decreases(self) -> None:
        """Test that training reduces loss over epochs."""
        from src.analysis.autoencoder import YieldCurveAutoencoder, train_autoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Create sample data with some structure
        np.random.seed(42)
        data = np.random.randn(200, input_dim).astype(np.float32)

        history = train_autoencoder(
            model=model,
            data=data,
            n_epochs=20,
            batch_size=32,
            learning_rate=0.01,
        )

        # Check that loss generally decreases
        first_half_mean = np.mean(history["train_loss"][:10])
        second_half_mean = np.mean(history["train_loss"][10:])

        assert second_half_mean < first_half_mean, (
            "Average loss in second half of training should be lower"
        )

    def test_train_autoencoder_with_validation(self) -> None:
        """Test training with validation data."""
        from src.analysis.autoencoder import YieldCurveAutoencoder, train_autoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        np.random.seed(42)
        train_data = np.random.randn(150, input_dim).astype(np.float32)
        val_data = np.random.randn(50, input_dim).astype(np.float32)

        history = train_autoencoder(
            model=model,
            data=train_data,
            n_epochs=5,
            batch_size=32,
            learning_rate=0.01,
            validation_data=val_data,
        )

        msg = "History should contain val_loss when validation data provided"
        assert "val_loss" in history, msg
        assert len(history["val_loss"]) == 5

    def test_train_autoencoder_reproducibility(self) -> None:
        """Test that training is reproducible with same seed."""
        from src.analysis.autoencoder import YieldCurveAutoencoder, train_autoencoder

        input_dim = 7
        latent_dim = 3

        np.random.seed(42)
        data = np.random.randn(100, input_dim).astype(np.float32)

        # Train first model
        torch.manual_seed(123)
        model1 = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        history1 = train_autoencoder(model1, data, n_epochs=5, batch_size=32)

        # Train second model with same seed
        torch.manual_seed(123)
        model2 = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        history2 = train_autoencoder(model2, data, n_epochs=5, batch_size=32)

        np.testing.assert_array_almost_equal(
            history1["train_loss"],
            history2["train_loss"],
            decimal=5,
            err_msg="Training should be reproducible with same seed",
        )


class TestCompareDimensionalityMethods:
    """Tests for comparing PCA vs autoencoder dimensionality reduction."""

    def test_compare_returns_results(self) -> None:
        """Test that comparison function returns expected structure."""
        from src.analysis.autoencoder import compare_dimensionality_methods
        from src.data.loader import InterestRateDataset

        # Create sample dataset
        np.random.seed(42)
        from datetime import date, timedelta

        start = date(2024, 1, 1)
        dates = [start + timedelta(days=i) for i in range(100)]
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        rates = np.random.randn(100, 7).astype(np.float64) * 0.01 + 0.03

        dataset = InterestRateDataset(
            file_path="test.csv",
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compare_dimensionality_methods(
            dataset=dataset,
            n_components=3,
            n_epochs=5,
        )

        # Check that result is a ComparisonResult dataclass with expected attributes
        assert hasattr(result, "pca_reconstruction_error")
        assert hasattr(result, "autoencoder_reconstruction_error")
        assert hasattr(result, "pca_explained_variance")
        assert hasattr(result, "autoencoder")  # The trained model

    def test_compare_pca_vs_autoencoder_errors(self) -> None:
        """Test that both methods produce finite reconstruction errors."""
        from src.analysis.autoencoder import compare_dimensionality_methods
        from src.data.loader import InterestRateDataset

        np.random.seed(42)
        from datetime import date, timedelta

        start = date(2024, 1, 1)
        dates = [start + timedelta(days=i) for i in range(100)]
        maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
        rates = np.random.randn(100, 7).astype(np.float64) * 0.01 + 0.03

        dataset = InterestRateDataset(
            file_path="test.csv",
            source_name="test",
            observation_dates=dates,
            maturities=maturities,
            rates=rates,
        )

        result = compare_dimensionality_methods(dataset, n_components=3, n_epochs=10)

        assert np.isfinite(result.pca_reconstruction_error)
        assert np.isfinite(result.autoencoder_reconstruction_error)
        assert result.pca_reconstruction_error >= 0
        assert result.autoencoder_reconstruction_error >= 0


class TestEncodeDecode:
    """Tests for encoding and decoding numpy arrays."""

    def test_encode_numpy_array(self) -> None:
        """Test encoding numpy arrays to latent space."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Numpy input
        data = np.random.randn(50, input_dim).astype(np.float32)

        # Get encodings
        encoded = model.encode_numpy(data)

        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (50, latent_dim)

    def test_decode_numpy_array(self) -> None:
        """Test decoding latent vectors to original space."""
        from src.analysis.autoencoder import YieldCurveAutoencoder

        input_dim = 7
        latent_dim = 3
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Latent vectors
        latent = np.random.randn(50, latent_dim).astype(np.float32)

        # Decode
        decoded = model.decode_numpy(latent)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (50, input_dim)

    def test_round_trip_numpy(self) -> None:
        """Test that encode-decode round trip produces similar output."""
        from src.analysis.autoencoder import YieldCurveAutoencoder, train_autoencoder

        input_dim = 7
        latent_dim = 5  # High enough to capture most variance
        model = YieldCurveAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Train on some data first
        np.random.seed(42)
        data = np.random.randn(200, input_dim).astype(np.float32)
        train_autoencoder(model, data, n_epochs=50, batch_size=32)

        # Test round trip
        test_data = np.random.randn(20, input_dim).astype(np.float32)
        encoded = model.encode_numpy(test_data)
        decoded = model.decode_numpy(encoded)

        # After training, reconstruction should be reasonably close
        mse = np.mean((test_data - decoded) ** 2)
        assert mse < 1.0, f"Reconstruction error too high: {mse}"
