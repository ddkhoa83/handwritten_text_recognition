"""Two-convolution MNIST CNN implemented with NumPy and manual backpropagation.

Architecture (default):
    (N,1,28,28) -> Conv16/ReLU/MaxPool -> Conv32/ReLU/MaxPool
    -> Flatten(1568) -> Dense128/ReLU -> Dense10/Softmax.

No PyTorch, TensorFlow, autograd, or pretrained layers are used.
The optimizer is an explicit Nesterov-momentum update.
"""

from __future__ import annotations

import argparse
import gzip
import math
import struct
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FeatureArray = NDArray[np.float32]
FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]
ParameterMap = dict[str, FloatArray]
ForwardCache = dict[str, object]
ConvCache = tuple[tuple[int, ...], FloatArray, FloatArray]
PoolCache = tuple[tuple[int, ...], LabelArray]


class CNN:
    """Two-convolution classifier with manually computed gradients.

    Input values must lie in [0, 1]. By default, training uses integer
    translation augmentation and validation-based early stopping.
    Calling fit() initializes new parameters and resets training histories.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 64,
        reg_strength: float = 0.0001,
        n_filters: int = 16,
        n_filters2: int = 32,
        hidden_units: int = 128,
        image_size: int = 28,
        shift_pixels: int = 2,
        patience: int = 6,
        lr_patience: int = 2,
        random_state: int = 42,
        verbose: bool = True,
        momentum: float = 0.9,
        n_classes: int = 10,
        in_channels: int = 1,
    ) -> None:
        for name, value in (("learning_rate", learning_rate),
                            ("reg_strength", reg_strength)):
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}: {value}.")
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1).")
        for name, value in (
            ("n_epochs", n_epochs), ("batch_size", batch_size),
            ("n_filters", n_filters), ("n_filters2", n_filters2),
            ("hidden_units", hidden_units), ("image_size", image_size),
            ("patience", patience), ("lr_patience", lr_patience),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four.")
        if type(shift_pixels) is not int or not 0 <= shift_pixels < image_size:
            raise ValueError("shift_pixels must be an integer in [0, image_size).")
        if type(random_state) is not int or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")
        if type(verbose) is not bool:
            raise ValueError("verbose must be a bool.")
        if type(n_classes) is not int or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2.")
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be a positive integer.")

        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_strength = float(reg_strength)
        self.n_filters = n_filters
        self.n_filters2 = n_filters2
        self.hidden_units = hidden_units
        self.image_size = image_size
        self.shift_pixels = shift_pixels
        self.patience = patience
        self.lr_patience = lr_patience
        self.random_state = random_state
        self.verbose = verbose
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_features_ = n_filters2 * (image_size // 4) ** 2

        self.parameters_: ParameterMap | None = None
        self.classes_: LabelArray | None = None
        self.loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        self.validation_accuracy_history_: list[float] = []
        self.best_epoch_: int | None = None

    def _initialize_parameters(self) -> None:
        """Initialize He-scaled weights and reset fitted statistics."""
        rng = np.random.default_rng(self.random_state)
        self.parameters_ = {}
        shapes = {
            "conv1": (self.n_filters, self.in_channels, 3, 3),
            "conv2": (self.n_filters2, self.n_filters, 3, 3),
            "hidden": (self.n_features_, self.hidden_units),
            "output": (self.hidden_units, self.n_classes),
        }
        for name, shape in shapes.items():
            fan_in = int(np.prod(shape[1:])) if name.startswith("conv") else shape[0]
            variance_factor = 1.0 if name == "output" else 2.0
            scale = math.sqrt(variance_factor / fan_in)
            self.parameters_[name + "_weights"] = rng.normal(0.0, scale, shape)
            width = shape[0] if name.startswith("conv") else shape[1]
            self.parameters_[name + "_bias"] = np.zeros(width, dtype=np.float64)

        self.classes_ = np.arange(self.n_classes, dtype=np.int64)
        self.loss_history_ = []
        self.validation_loss_history_ = []
        self.validation_accuracy_history_ = []
        self.best_epoch_ = None

    def _check_is_fitted(self) -> None:
        """Reject inference before fit() has initialized the model."""
        if self.parameters_ is None or self.classes_ is None:
            raise RuntimeError("Call fit() before using this method.")

    def _validate_features(self, X: FeatureArray | FloatArray) -> np.ndarray:
        """Return a view of images with shape (N,C,H,W), normalized to [0,1]."""
        values = np.asarray(X)
        width, channels = self.image_size, self.in_channels
        if values.ndim == 2 and values.shape[1] == channels * width * width:
            values = values.reshape(-1, channels, width, width)
        elif values.ndim == 3 and values.shape[1:] == (width, width):
            if channels != 1:
                raise ValueError("(N,H,W) is only valid for single-channel images.")
            values = values[:, None, :, :]

        if values.ndim != 4 or values.shape[1:] != (channels, width, width) or len(values) == 0:
            raise ValueError(
                f"X must have shape (N,{channels * width * width}) or "
                f"(N,{channels},{width},{width}); "
                "(N,H,W) is also accepted for single-channel input."
            )
        if values.dtype.kind not in "fiu":
            raise ValueError("X must contain real numeric pixels.")
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start + self.batch_size]
            if not np.isfinite(batch).all() or np.any(batch < 0) or np.any(batch > 1):
                raise ValueError("Pixels must be finite and normalized to [0, 1].")
        return values

    def _validate_labels(self, y: LabelArray, count: int) -> LabelArray:
        """Validate a one-dimensional array of integer class indices."""
        labels = np.asarray(y)
        if labels.shape != (count,) or labels.dtype.kind not in "iu":
            raise ValueError("y must be a one-dimensional integer array matching X.")
        if np.any(labels < 0) or np.any(labels >= self.n_classes):
            raise ValueError(f"Labels must be in [0, {self.n_classes - 1}].")
        return labels.astype(np.int64, copy=False)

    @staticmethod
    def _log_softmax(scores: FloatArray) -> FloatArray:
        """Compute stable log softmax across classes (axis=1)."""
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite logits; reduce learning_rate.")
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    @staticmethod
    def _conv_forward(X: FloatArray, W: FloatArray, b: FloatArray) -> tuple[FloatArray, ConvCache]:
        """3x3 zero-padded cross-correlation: (B,C,H,W) -> (B,F,H,W)."""
        count, channels, height, width = X.shape
        padded = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
        patches = np.empty((count, height, width, channels, 3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                patches[:, :, :, :, row, column] = padded[
                    :, :, row:row + height, column:column + width
                ].transpose(0, 2, 3, 1)
        columns = patches.reshape(count * height * width, channels * 9)
        output = columns @ W.reshape(len(W), -1).T + b
        logits = output.reshape(count, height, width, len(W)).transpose(0, 3, 1, 2)
        return logits, (X.shape, columns, W)

    @staticmethod
    def _conv_backward(
        gradient: FloatArray,
        cache: ConvCache,
        need_input: bool = True,
    ) -> tuple[FloatArray | None, FloatArray, FloatArray]:
        """Manually compute (dX,dW,db) for the padded 3x3 convolution.

        dW and db are summed over the batch and spatial coordinates. The
        incoming gradient is already normalized by batch size (at the loss).
        """
        shape, columns, W = cache
        count, channels, height, width = shape
        rows = gradient.transpose(0, 2, 3, 1).reshape(-1, len(W))
        dW = (rows.T @ columns).reshape(W.shape)
        db = rows.sum(axis=0)
        if not need_input:
            return None, dW, db

        # Each output gradient contributes to a 3x3 input patch. These patches
        # overlap, so accumulate their contributions into a *separate* padded dX.
        patch_gradient = (rows @ W.reshape(len(W), -1)).reshape(
            count, height, width, channels, 3, 3
        )
        padded_gradient = np.zeros(
            (count, channels, height + 2, width + 2), dtype=np.float64
        )
        for row in range(3):
            for column in range(3):
                padded_gradient[
                    :, :, row:row + height, column:column + width
                ] += patch_gradient[:, :, :, :, row, column].transpose(0, 3, 1, 2)
        dX = padded_gradient[:, :, 1:-1, 1:-1]
        return dX, dW, db

    @staticmethod
    def _pool_forward(X: FloatArray) -> tuple[FloatArray, PoolCache]:
        """MaxPool(2x2,stride=2), retaining argmax switches for backward."""
        count, channels, height, width = X.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Max pooling needs even image dimensions.")
        windows = X.reshape(count, channels, height // 2, 2, width // 2, 2)
        windows = windows.transpose(0, 1, 2, 4, 3, 5).reshape(
            count, channels, height // 2, width // 2, 4
        )
        return windows.max(axis=-1), (X.shape, windows.argmax(axis=-1))

    @staticmethod
    def _pool_backward(gradient: FloatArray, cache: PoolCache) -> FloatArray:
        """Send each output gradient to only the stored maximum location."""
        shape, switches = cache
        count, channels, height, width = shape
        windows = np.zeros((*gradient.shape, 4), dtype=np.float64)
        np.put_along_axis(windows, switches[..., None], gradient[..., None], axis=-1)
        return windows.reshape(
            count, channels, height // 2, width // 2, 2, 2
        ).transpose(0, 1, 2, 4, 3, 5).reshape(shape)

    @staticmethod
    def _translate(X: FloatArray, rng: np.random.Generator, shift_pixels: int) -> FloatArray:
        """Shift each (C,H,W) training image by random integer offsets.

        Pixels outside the image are dropped; uncovered pixels are set to zero.
        No wrap-around or interpolation is performed.
        """
        if shift_pixels == 0:
            return X
        count, _, height, width = X.shape
        result = np.zeros_like(X)
        offsets = rng.integers(-shift_pixels, shift_pixels + 1, size=(count, 2))
        for index, (dy, dx) in enumerate(offsets):
            source_y, source_x = max(0, -dy), max(0, -dx)
            target_y, target_x = max(0, dy), max(0, dx)
            overlap_h, overlap_w = height - abs(dy), width - abs(dx)
            result[
                index, :, target_y:target_y + overlap_h, target_x:target_x + overlap_w
            ] = X[index, :, source_y:source_y + overlap_h, source_x:source_x + overlap_w]
        return result

    def _forward(self, X: FloatArray) -> tuple[FloatArray, ForwardCache]:
        """Evaluate network logits and collect caches for manual backward."""
        self._check_is_fitted()
        assert self.parameters_ is not None
        p = self.parameters_
        z1, conv1 = self._conv_forward(X, p["conv1_weights"], p["conv1_bias"])
        a1, pool1 = self._pool_forward(np.maximum(z1, 0))
        z2, conv2 = self._conv_forward(a1, p["conv2_weights"], p["conv2_bias"])
        a2, pool2 = self._pool_forward(np.maximum(z2, 0))
        flat = a2.reshape(len(X), -1)
        hidden_z = flat @ p["hidden_weights"] + p["hidden_bias"]
        hidden = np.maximum(hidden_z, 0)
        scores = hidden @ p["output_weights"] + p["output_bias"]
        cache: ForwardCache = {
            "z1": z1, "z2": z2, "conv1": conv1, "conv2": conv2,
            "pool1": pool1, "pool2": pool2, "flat": flat,
            "pooled_shape": a2.shape, "hidden_z": hidden_z, "hidden": hidden,
        }
        return scores, cache

    def _backward(self, score_gradient: FloatArray, cache: ForwardCache) -> ParameterMap:
        """Backpropagate normalized loss derivatives through every layer."""
        assert self.parameters_ is not None
        p, c = self.parameters_, cache
        gradients: ParameterMap = {}
        gradients["output_weights"] = c["hidden"].T @ score_gradient
        gradients["output_bias"] = score_gradient.sum(axis=0)
        hidden_gradient = (score_gradient @ p["output_weights"].T) * (c["hidden_z"] > 0)
        gradients["hidden_weights"] = c["flat"].T @ hidden_gradient
        gradients["hidden_bias"] = hidden_gradient.sum(axis=0)
        pooled_gradient = (hidden_gradient @ p["hidden_weights"].T).reshape(c["pooled_shape"])
        z2_gradient = self._pool_backward(pooled_gradient, c["pool2"]) * (c["z2"] > 0)
        a1_gradient, gradients["conv2_weights"], gradients["conv2_bias"] = (
            self._conv_backward(z2_gradient, c["conv2"])
        )
        assert a1_gradient is not None
        z1_gradient = self._pool_backward(a1_gradient, c["pool1"]) * (c["z1"] > 0)
        _, gradients["conv1_weights"], gradients["conv1_bias"] = self._conv_backward(
            z1_gradient, c["conv1"], need_input=False
        )
        return gradients

    def _regularization_loss(self) -> float:
        """Return 0.5 * lambda * sum(weight**2), excluding biases."""
        assert self.parameters_ is not None
        return 0.5 * self.reg_strength * sum(
            float(np.sum(value ** 2))
            for name, value in self.parameters_.items() if name.endswith("_weights")
        )

    def _loss_and_gradients(self, X: FloatArray, y: LabelArray) -> tuple[float, ParameterMap]:
        """Compute mini-batch cross-entropy + L2 loss and all gradients."""
        scores, cache = self._forward(X)
        log_probabilities = self._log_softmax(scores)
        count = len(X)
        data_loss = -float(np.mean(log_probabilities[np.arange(count), y]))
        errors = np.exp(log_probabilities)
        errors[np.arange(count), y] -= 1.0
        errors /= count
        gradients = self._backward(errors, cache)
        assert self.parameters_ is not None
        for name, weights in self.parameters_.items():
            if name.endswith("_weights"):
                gradients[name] += self.reg_strength * weights
        return data_loss + self._regularization_loss(), gradients

    def _nag_update(self, gradients: ParameterMap, velocity: ParameterMap, rate: float) -> None:
        """Apply Nesterov momentum with the PyTorch-style buffer convention.

        buffer = momentum * buffer + grad
        correction = grad + momentum * buffer
        parameter = parameter - rate * correction
        """
        assert self.parameters_ is not None
        for name, gradient in gradients.items():
            velocity[name] = self.momentum * velocity[name] + gradient
            update = gradient + self.momentum * velocity[name]
            self.parameters_[name] -= rate * update
            if not np.isfinite(self.parameters_[name]).all():
                raise FloatingPointError("Training diverged; lower learning_rate.")

    def _evaluate(self, X: np.ndarray, y: LabelArray) -> tuple[float, float]:
        """Return validation cross-entropy and accuracy without augmentation."""
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), self.batch_size):
            batch = np.asarray(X[start:start + self.batch_size], dtype=np.float64)
            labels = y[start:start + self.batch_size]
            scores, _ = self._forward(batch)
            logp = self._log_softmax(scores)
            total_loss -= float(logp[np.arange(len(batch)), labels].sum())
            correct += int(np.sum(scores.argmax(axis=1) == labels))
        return total_loss / len(X), correct / len(X)

    def fit(
        self,
        X: FeatureArray | FloatArray,
        y: LabelArray,
        X_val: FeatureArray | FloatArray | None = None,
        y_val: LabelArray | None = None,
    ) -> CNN:
        """Train on mini-batches; optionally select best validation checkpoint.

        Validation inputs must be disjoint from both training and test data.
        Without validation, train for all epochs and keep the final weights.
        """
        images = self._validate_features(X)
        labels = self._validate_labels(y, len(images))
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val, or neither.")
        validation = self._validate_features(X_val) if X_val is not None else None
        targets = self._validate_labels(y_val, len(validation)) if validation is not None else None

        self._initialize_parameters()
        assert self.parameters_ is not None
        rng = np.random.default_rng(self.random_state)
        velocity = {name: np.zeros_like(value) for name, value in self.parameters_.items()}
        stale, lr_stale = 0, 0
        rate = self.learning_rate
        best_key = (-1.0, -float("inf"))
        best_parameters: ParameterMap | None = None

        for epoch in range(self.n_epochs):
            started = time.perf_counter()
            order = rng.permutation(len(images))
            total_loss = 0.0
            for start in range(0, len(images), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = np.asarray(images[indices], dtype=np.float64)
                batch = self._translate(batch, rng, self.shift_pixels)
                loss, gradients = self._loss_and_gradients(batch, labels[indices])
                if not math.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; lower learning_rate.")
                total_loss += loss * len(indices)
                # IMPORTANT: update parameters for *each* mini-batch.
                self._nag_update(gradients, velocity, rate)

            self.loss_history_.append(total_loss / len(images))
            message = (
                f"Epoch {epoch + 1:03d}/{self.n_epochs:03d} | "
                f"Train objective (online): {self.loss_history_[-1]:.6f} | "
                f"LR: {rate:.6g}"
            )

            if validation is not None:
                assert targets is not None
                val_loss, val_accuracy = self._evaluate(validation, targets)
                self.validation_loss_history_.append(val_loss)
                self.validation_accuracy_history_.append(val_accuracy)
                message += f" | Val loss: {val_loss:.6f} | Val accuracy: {val_accuracy:.4%}"

                key = (val_accuracy, -val_loss)
                if key > best_key:
                    best_key = key
                    best_parameters = {
                        name: value.copy() for name, value in self.parameters_.items()
                    }
                    self.best_epoch_ = epoch + 1
                    stale = 0
                    lr_stale = 0
                else:
                    stale += 1
                    lr_stale += 1
                    if lr_stale >= self.lr_patience:
                        rate = max(rate * 0.5, self.learning_rate * 0.01)
                        lr_stale = 0

            if self.verbose:
                print(message + f" | {time.perf_counter() - started:.1f}s", flush=True)
            if validation is not None and stale >= self.patience:
                if self.verbose:
                    print("Early stopping: no validation checkpoint improvement.", flush=True)
                break

        if best_parameters is not None:
            self.parameters_ = best_parameters
            if self.verbose:
                print(
                    f"Restored epoch {self.best_epoch_}; "
                    f"validation accuracy: {best_key[0]:.4%}", flush=True
                )
        else:
            self.best_epoch_ = len(self.loss_history_)
        return self

    def predict_proba(self, X: FeatureArray | FloatArray) -> FloatArray:
        """Return class probabilities of shape (N,n_classes)."""
        self._check_is_fitted()
        images = self._validate_features(X)
        result = np.empty((len(images), self.n_classes), dtype=np.float64)
        for start in range(0, len(images), self.batch_size):
            batch = np.asarray(images[start:start + self.batch_size], dtype=np.float64)
            scores, _ = self._forward(batch)
            result[start:start + len(batch)] = np.exp(self._log_softmax(scores))
        return result

    def predict(self, X: FeatureArray | FloatArray) -> LabelArray:
        """Return the most probable class for each image."""
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    def score(self, X: FeatureArray | FloatArray, y: LabelArray) -> float:
        """Return classification accuracy as a fraction in [0,1]."""
        predicted = self.predict(X)
        labels = self._validate_labels(y, len(predicted))
        return float(np.mean(predicted == labels))

