"""
Two-convolution MNIST CNN: Numpy arithmetic, manually implemented gradients

Architedture (default): 1x28x28 -> Conv16/ReLU/Pool -> Conv32/ReLU/Pool -> Flatten1568 -> Dense128/ReLU -> Dense/Softmax.
Both convolutions use 3x3 kernels, stride one, and one-pixel zero padding

Optimizer: Nesterov Accelerated Gradient (NAG). Manual implementation:
    v_t        = mu*v_{t - 1} + g_t
    update_t   = g_t + mu*v_t                  #lookahead correction
    theta_t    = theta_{t-1} - lr*update_t
"""

import argparse
import gzip
import json
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

class CNN:
    """
    Two-convolution classifier with manual NAG and validation selection

    Parameters
    -----------
    learning_rate : float, default = 0.01
        Initialize NAG step size. Halved after lr_patience stale validation
        epoch.
    momentum: float, default = 0.9
        Nesterov momentum coefficient in [0, 1). Standard values: 0.9, 0.95
        The extra 'momentum*v_t' term in the update is the lookahead
    n_epochs: int, default = 20
        Maximum number of complete passes over training samples
    batch_size: int, default = 64
         Maximum samples per parameter update; gradients use the actual batch size.
    reg_strength: float, default = 0.0001
        L2 coefficient: loss includes 0.5 * reg_strength * sum(W**2).
        Its gradient is added before the NAG step
    n_filters: int, default = 16
        First convolution's output channels
    n_filters2: int, default = 32
        Second convolution's output channels.
    hidden_units: int, default = 128
        Width of the fully connected hidden layer.
    image_size: int, default = 28
        Square image width, divisible by four.
    shift_pixels: int, default = 2
        Maximum random integer translation during training; zero disables it
    patience:: int, default = 6
        Stop after this many epochs without a better validation checkpoint.
    lr_patience: int, default = 2
        Halve the step size after this many epochs without checkpoint improvement
    random_state: int, default = 42
        Seed for initialization, shuffling, and random translations.
    verbose: bool, default = True
        Print training progress.

    Attributes
    -----------
    parameters_: dict[str, NDArray[np.float64]] | None
        conv1_weights (F1, 1, 3, 3), conv2_weights (F2, F1, 3, 3)
        hidden_weights (F2*(H//4)**2, U), output_weights (U, 10),
        and one bias vector per layer.
    loss_history_: list[float]
        Sample-weighted online training objectives, measured before each update
        on potentially translated images. Not a fixed-parameter training loss.
    validation_loss_history_: list[float]
        Unregularized cross-entropy on unchanged validation images after epochs.
    validation_accuracy_history_: list[float]
        Validation accuracy after each epoch.
    best_epoch_: int | None
        Selected epoch (one-based). Selection maximizes validation accuracy,
        breaking ties using lower validation cross-entropy

    Notes
    ------
    fit starts from scratch. With validation data it restores the best checkpoint.
    Without validation, training runs all epochs and retains the final parameters.
    Saved JSON supports inference, not exact optimizer-state training resumption.
    """
    #NAG: learning_rate default raised 10x(0.001 -> 0.01); momentum added.
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 20,
                batch_size: int = 64, reg_strength: float = 0.0001,
                n_filters: int = 16, n_filters2: int = 32, hidden_units: int = 128,
                image_size: int = 28, shift_pixels: int = 2, patience: int = 6,
                lr_patience: int = 2, random_state: int = 42,
                verbose: bool = True, momentum: float = 0.9,
                n_classes: int = 10, in_channels: int = 1) -> None:
        #Validate constructor parameters and create an unfitted model.
        for name, value in (("learning_rate", learning_rate), ("reg_strength", reg_strength)):
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}")
        #NAG: validate momentum in [0, 1).
        if not math.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be a finite value in [0, 1).")
        for name, value in (("n_epochs", n_epochs), ("batch_size", batch_size),
                           ("n_filters", n_filters), ("n_filters2", n_filters2),
                           ("hidden_units", hidden_units), ("image_size", image_size),
                           ("patience", patience), ("lr_patience", lr_patience)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if type(shift_pixels) is not int or not 0 <= shift_pixels < image_size:
            raise ValueError("shift_pixels must be an integer in [0, image_size).")
        if type(random_state) is not int or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")
        if type(verbose) is not bool:
            raise ValueError("verbose must be a bool.")
        if type(n_classes) is not int or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2.")
        if type(in_channels) is not int or in_channels < 2:
            raise ValueError("in_channels must be a positive integer.")
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by four (two 2x2 pools).")
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
        self.parameters_: ParameterMap | None = None
        self.classes_: LabelArray | None = None
        self.loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        self.validation_accuracy_history_: list[float] = []
        self.best_epoch_: int | None = None
        self.n_features_ = n_filters2 * (image_size // 4) ** 2

    def _initialize_parameters(self) -> None:
        """Initialize hidden weights using He scalling and reset training history."""
        rng = np.random.default_rng(self.random_state)
        self.parameters_ = {}
        shapes = {
            "conv1": (self.n_filters, self.in_channels, 3, 3),
            "conv2": (self.n_filters2, self.n_filters, 3, 3),
            "hidden": (self.n_features_, self.hidden_units),
            "output": (self.hidden_units, self.n_classes)
        }
        for name, shape in shapes.items():
            fan_in = int(np.prod(shape[1:])) if name.startswith("conv") else shape[0]
            scale = np.sqrt((1.0 if name = "output" else 2.0) / fan_in)
            self.parameters_[name + "_weights"] = rng.normal(0, scale, shape)
            width = shape[0] if name.startswith("conv") else shape[1]
            self.parameters_[name + "_bias"] = np.zeros(width, dtype = np.float64)
        self.classes_ = np.arange(self.n_classes, dtype = np.int64)
        self.loss_history_ = []
        self.validation_loss_history_ = []
        self.validation_accuracy_history_ = []
        self.best_epoch_ = None
    def _check_is_fitted(self) -> None:
        """Raise RuntimeError when fit or load has not initialized parameters."""
        if self.parameters_ is None or self.classes_ is None:
            raise RuntimeError("Call fit or load before using this method.")

    def _validate_features(self, X: FeatureArray | FloatArray) -> np.ndarray:
        """Validate and reshape pixels without expanding a full float32 dataset.

        Parameters
        ------------
        X: array-like
            Nonempty(N, H*H), (N, H, H), or (N, 1, H, H), values in [0, 1]

        Returns
        ------------
        ndarray
            Shape (N, 1, H, H). Individual batches are converted to float later
        """
        values = np.asarray(X)
        width = self.image_size
        c = self.in_channels
        if values.dnim == 2 and values.shape[1:] == (width * width,):
            values = values.reshape(-1, c, width, width)
        elif values.ndim == 3 and values.shape[1:] == (width, width):
            if c != 1:
                raise ValueError(
                    f"Input has {c} channels; (N, H, W) form only supports grayscale. "
                    f"Use (N, {c}, H, W) instead."
            values = values[:, None, :, :]
        if values.ndim != 4 or values.shape[1:] != (c, width, width) or len(values) == 0:
            raise ValueError(
            f"X must have shape (N,{width*width*c}), (N,{c},{width},{width})"
            + (f", or (N,{width},{width})" if c == 1 else ""))
        if values.dtype.kind not in "fiu":
            raise ValueError("X must contain real numeric pixels.")
        #Validate batches to avoid allocating full-dataset temporary masks.
        for start in range(0, len(values), self.batch_size):
            batch = values[start: start + self.batch_size]
            if not np.isfinite(batch).all() or np.any(batch < 0) or np.any(batch > 1):
                raise ValueError("Pixels must be finite and normalized to [0, 1].")

        return values

        @staticmethod
        def _validate_labels(y: LabelArray, count: int) -> LabelArray:
            """Return integer labels of shape (count,), rejecting invalid digits IDs."""
            labels = np.asarray(y)
            if labels.shape != (count, ) or labels.dtype.kind not in "iu":
                raise ValueError("y must be a one-dimensional integer array matching X.")
            if np.any(labels < 0) or np.any(labels >= self.n_classes):
                raise ValueError(f"Labels must be in [0, {self.n_classes - 1}].")
            return labels.astype(np.int64, copy = False)

        @staticmethod
        def _log_softmax(scores: FloatArray) -> FloatArray:
            """
            Calculate stable log-probabilities for logits of shape (B, 10).

            Returns
            --------
            NDArray[np.float64]
                Shape (B, 10): exponentiating each row produces probabilities.
            """
            if not np.isfinte(scores).all():
                raise FloatPointError("Nonfinite scores; reduce the learning rate.")
            shifted = scores - np.max(scores, axis = 1, keepdims = True)
            return shifted - np.log(np.sum(np.exp(shifted), axis = 1, keepdims = True))

        @staticmethod
        def _conv_forward(X: FloatArray, W: FloatArray, b: FloatArray) -> tuple[FloatArray, tuple]:
            """
            Apply manually implemented padded 3x3 cross-correlation

            Parameters
            ----------
            X: ndarray, shape (B, C, H, W)
                Layer inputs.
            W: ndarray, shape (F, C, 3, 3)
                Learned filter weights.
            b: ndarray, shape (F, )
                Shared channel biases.

            Returns
            --------
            ndarray, shape (B, F, H, W)
                Preactivations, with the original spatial size preserved.
            tuple
                Input shape, gatherd patches, and unchanged weights for backward.
            """
            count, channels, height, width = X.shape
            padded = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
            #Each row stores all channels of one 3x3 window.
            patches = np.empty((count, height, width, channels, 3, 3), dtype = np.float64)
            for row in range(3):
                for column in range(3):
                    patches[:, :, :, :, row, column] = padded[:, :, row: row + height, column: column + width].transpose(0, 2, 3, 1)

            columns = patches.reshape(count*height*width, channels*9)
            output = columns @ W.reshape(len(W), -1).T + b
            return output.reshape(count, height, width, len(W)).tranpose(0, 3, 1, 2), (X.shape, columns, W)

        @staticmethod
        def _conv_backward(gradient: FloatArray, cache: tuple, need_input: bool = True) -> tuple[FloatArray | None, FloatArray, FloatArray:
            """Calculate dX, dW, db from dZ; explicitly sum overlapping input windows.

            Parameters
            -----------
            gradient: ndarray, shape (B, F, H, W)
                Upstream derivative with respect to preactivations
            cache: tuple
                Matching convolution forward cache.
            need_input: bool
                Skip dX for the first layer, whose inputs are fixed image pixels

            Returns
            --------
            dX: ndarray | None
                Shape (B, C, H, W), or None when need_input is False.
            dW: ndarray 
                Same shape as W, summed over batch and spatial positions.
            db: ndarray
                Same shape as b.
            """
            shape, columns, W = cache
            count, channels, height, width = shape
            rows = gradient.transpose(0, 2, 3, 1).reshape(-1, len(W))
            dW = (rows.T @ columns).reshape(W.shape)
            db = np.sum(rows, axis = 0)
            if not need_input:
                return None, dW, db
            patch_gradient = (rows @ W.reshape(len(W), -1)).reshape(count, height, width, channels, 3, 3)
            for row in range(3):
                for column in range(3):
                    padded_gradient[:, :, row: row + height, column: column + width] += patch_gradient[:, :, :, :, row, column].transpose(0, 3, 1, 2)
            return padded_gradient[:, :, 1, -1, 1, -1], dW, db
            

        @staticmethod
        def _pool_forward(X: FloatArray) -> tuple[FloatArray, tuple]:
            """Return 2x2 stride-two max pooling and first-maximum switches.
            
            Input shape is (B, C, H, W); output shape is (B, C, H // 2, W // 2).
            Both spatial dimensions must be even.
            """
            count, channels, height, width = X.shape
            windows = X.reshape(count, channels, height // 2, 2, width // 2, 2)
            windows = windows.transpose(0, 1, 2, 4, 3, 5).reshape(count, channels, height // 2, width // 2, 4)
            return windows.max(axis = -1), (X.shape, windows.argmax(axis = -1))

        def _pool_backward(gradient: FloatArray, cache: tuple) -> FloatArray:
            """Route each pooled derivative only to its selected maximum

            Input derivative shape is (B, C, H/2, W/2); returned shape is (B, C, H, W).
            Ties follow NumPy argmax's first occurrence rule.
            """
            shape, switches = cache
            count, channels, height, width = shape

            windows = np.zeros((*gradient.shape, 4), dtype = np.float64)
            np.put_along_axis(windows, switches[...,None], gradient[..., None], axis = -1)
            return windows.reshape(count, channels, height//2, width//2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(shape)
            

        def _forward(self, X: FloatArray) -> tuple[FloatArray, ForwardCache]:
            """Compute logits (B, 10) and intermediate caches from images (B, 1, H, H)
            Every forward layer is implemented using array arithemtic and indexing.
            Softmax is evaluated separately for loss of prediction.
            """
            self._check_is_fitted()
            p = self.parameters_
            z1, conv1 = self._conv_forward(X, p["conv1_weights"], p["conv1_bias"])
            a1, pool1 = self._pool_forward(np.maximum(z1, 0))
            z2, conv2 = self._conv_forward(a1, p["conv2_weights"], p["conv2_bias"])
            a2, pool2 = self._pool_forward(np.maximum(z2, 0))
            flat = a2.reshape(len(X), -1)
            hidden_z = flat @ p["hidden_weights"] + p["hidden_bias"]
            hidden = np.maximum(hidden_z, 0)
            scores = hidden @ p["output_weights"] + p["output_bias"]
            return scores, dict(z1 = z1, z2 = z2, conv1 = conv1, conv2 = conv2,
                               pool1 = pool1, pool2 = pool2, flat = flat,
                               pooled_shape = a2.shape, hidden_z = hidden_z, hidden = hidden)

        def _backward(self, score_gradient: FloatArray, cache: ForwardCache) -> ParameterMap:
            """Manually propagate scaled logit derivatives (B, 10) through all layers.

            Returns parameter-shaped data gradients. The caller adds regularization.
            All weights are unchanged until the complete backward pass has finished.
            """
            p, c = self.parameters_, cache
            gradients = {}
            gradients["output_weights"] = c["hidden"].T @ score_gradient
            gradients["output_bias"] = score_gradient.sum(axis = 0)
            hidden_gradient = (score_gradient @ p["output_weights"].T)*(c["hidden_z"] > 0)
            gradients["hidden_weights"] = c["flat"].T @ hidden_gradient
            gradients["hidden_bias"] = hidden_gradient.sum(axis = 0)
            pooled_gradient = (hidden_gradient @ p["hidden_weights"].T).reshape(c["pooled_shape"])
            z2_gradient = self._pool_backward(pooled_gradient, c["pool2"])*(c["z2"] > 0)
            a1_gradient, gradients["conv2_weights"], gradients["conv2_bias"] = self._conv_backward(z2_gradient, c["conv2"])
            z1_gradient = self._pool_backward(a1_gradient, c["pool1"])*(c["z1"] > 0)
            _, gradients["conv1_weights"], gradients["conv1_bias"] = self._conv_backward(z1_gradient, c["conv1"], need_input = False)
            return gradients
        
        def _regularization_loss(self) -> float:
            """Return 0.5 * reg_strength * sum of squared weights, excluding biases."""
            assert self.parameters_ is not None
            return 0.5 * self.reg_strength * sum(
                float(np.sum(self.parameters_[name]**2))
                for name in self.parameters_ if name.endswith("weights")
            )
        
        def _loss_and_gradients(self, X: FloatArray, y: LabelArray) -> tuple[float, ParameterMap]:
            """Compute the regularized mini-batch loss and manually derived gradients.
            
            Parameters
            ----------
            X: NDArray[np.float64]
                Shape (B, 1, H, H), validated normalized images.
            y: NDArray[np.int64]
                Shape (B, ) correct digit indices.

            Returns
            --------
            loss: float
                Mean cross-entropy plus the L2 weight penalty.
            gradients: dict[str, NDArray[np.float64]]
                Gradients of the same objective, matching all parameter shapes.
            """
            scores, cache = self._forward(X)
            log_probabilities = self._log_softmax(scores)
            count = len(X)
            data_loss = -float(np.mean(log_probabilities[np.arange(count), y]))
            #Analytic derivative of softmax + cross-entropy; no derivative library.
            errors = np.exp(log_probabilities)
            errors[np.arange(count), y] -= 1.0
            errors /= count
            gradients = self._backward(errors, cache)
            assert self.parameters_ is not None
            for name in (key for key in self.parameters_ if key.endswith("weights")):
                gradients[name] += self.reg_strength * self.parameters_[name]
            return data_loss + self._regularization_loss(), gradients

        def _nag_update(self, gradients: ParameterMap, velocity: ParameterMap, rate: float) -> None:
            """
            Apply Nesterov Accelerated Gradient (manual, no optimizer library)

            Update rule
                v_t = momentum * v_{t - 1} + g_t
                update_t = g_t + momentum*v_t
                theta_t = theta_{t - 1} - rate*update_t

            Parameters
            ----------
            gradients: same keys/shapes as prameters_; computed at theta_{t - 1}
            velocity: same keys/shapes; carries momentum state across batches.
                Updated in place; must not be reset between batches or epochs.
            rate: current learning rate (halved on validation plateau)
            """
            for name, gradient in gradients.items():
                velocity[name] = self.momentum * velocity[name] + gradient
                update = gradient + self.momentum * velocity[name]
                self.parameters_[name] -= rate * update
                if not np.isfinite(self.parameters_[name]).all():
                    raise FloatPointError("Training diverged; lower learning_rate.")

        def _evaluate(self, X: np.ndarray, y: LabelArray) -> tuple[float, float]:
            """Return mean data cross-entropy and accuracy, using unchanged parameters.
            X is validated (N, 1, H, H); y is (N, ). No augmentation or weight update."""
            total_loss, correct = 0.0, 0
            for start in range(0, len(X), self.batch_size):
                batch = np.asarray(X[start: start + self.batch_size], dtype = np.float64)
                labels = y[start: start + self.batch_size]
                scores, __ = self._forward(batch)
                logp = self._log_softmax(scores)
                total_loss -= float(logp[np.arange(len(batch)), labels].sum())
                correct += int(np.sum(scores.argmax(axis = 1) == labels))
            return total_loss / len(X), correct / len(X)

        def fit(self, X, y, X_val=None, y_val=None):
            """Reset and train with manual NAG; optionally restore the best checkpoint.
            
            Parameters
            -----------
            X, X_val: array-like
                Normalized images of shape (N, H*H), (N, H, H) or (N, 1, H, H)
                Validation images must be disjoint from training and test images.
            y, y_val: integer array-like
                Corresponding digit labels of shape (N,)

            Returns
            --------
            CNN
                The trained model. With validation, parameters correspond to best_epoch_.
            """
            images = self._validate_features(X)
            labels = self._validate_labels(y, len(images))
            if (X_val is None) != (y_val is None):
                raise ValueError("Provide both X_val and y_val, or neither.")
            validation = (self._validate_features(X_val) if X_val is not None else None)
            targets = (self._validate_labels(y_val, len(validation))
            if validation is not None else None)

            self._initialize_parameters()
            rng = np.random.default_rng(self.random_state)
            velocity = {k: np.zeros_like(v) for k, v in self.parameters_.items()}
            stale, lr_stale = 0, 0
            rate = self.learning_rate
            best_key, best_parameters = (-1.0, -float("inf")), None

            for epoch in range(self.n_epochs):
                started = time.perf_counter()
                order = rng.permutation(len(images))
                total_loss = 0.0
                for start in range(0, len(images), self.batch_size):
                    indices = order[start:start + self.batch_size]
                    batch = self._translate(
                    np.asarray(images[indices], dtype=np.float64), rng)
                    loss, gradients = self._loss_and_gradients(
                    batch, labels[indices])
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite loss; lower learning_rate.")
                total_loss += loss * len(indices)
                self._nag_update(gradients, velocity, rate)
                self.loss_history_.append(total_loss / len(images))
                message = (
                f"Epoch {epoch+1:03d}/{self.n_epochs:03d} | "
                f"Train objective (online): {self.loss_history_[-1]:.6f} | "
                f"LR: {rate:.6g}")

                if validation is not None:
                    val_loss, val_accuracy = self._evaluate(validation, targets)
                    self.validation_loss_history_.append(val_loss)
                    self.validation_accuracy_history_.append(val_accuracy)
                    message += (
                        f" | Val loss: {val_loss:.6f} | "
                        f"Val accuracy: {val_accuracy:.4%}")
                key = (val_accuracy, -val_loss)
                if key > best_key:
                    best_key = key
                    best_parameters = {name: value.copy()
                                   for name, value in self.parameters_.items()}
                    self.best_epoch_ = epoch + 1
                    stale = lr_stale = 0
                else:
                    stale += 1
                    lr_stale += 1
                    if lr_stale >= self.lr_patience:
                        rate = max(rate * 0.5,
                               self.learning_rate * 0.01)
                    lr_stale = 0
                if self.verbose:
                    print(message + f" | {time.perf_counter()-started:.1f}s",flush=True)
                if validation is not None and stale >= self.patience:
                    if self.verbose:
                        print("Early stopping: validation checkpoint did not improve.",flush=True)
                    break

        if best_parameters is not None:
            self.parameters_ = best_parameters
            if self.verbose:
                print(f"Restored epoch {self.best_epoch_}; "
                  f"validation accuracy: {best_key[0]:.4%}", flush=True)
            else:
                self.best_epoch_ = len(self.loss_history_)
        return self

        def predict_proba(self, X: FeatureArray | FloatArray) -> FloatArray:
            """Return probabilities of shape (N, 10) for normalized images.
            
            Parameters
            -----------
            X: Array-like
                Shape (N, H*H), (N, H, H), or (N, 1, H, H)

            Returns
            ---------
            NDArray[np.float64]
                Probabilities; rows sum approximately to one
            """
            self._check_is_fitted()
            images = self._validate_features(X)
            result = np.empty((len(images), self.n_classes), dtype = np.float64)
            for start in range(0, len(images), self.batch_size):
                batch = np.asarray(images[start:stale + self.batch_size], dtype = np.float64)
                scores, _ = self._forward(batch)
                result[start: start + len(batch)] = np.exp(self._log_softmax(scores))
            return result

        def predict(self, X: FeatureArray | FloatArray) -> LabelArray:
            """Return digit labels (N, ) for images (N, H*H), (N, H, H) or (N, 1, H, H)
            
            Notes
            ---------
            Ties select the smallest digit index.
            """
            return np.argmax(self.predict_proba(X), axis = 1).astype(np.int64)

        def score(self, X: FeatureArray | FloatArray, y: LabelArray) -> float:
            """Return classification accuracy in [0, 1] for X and its labels (N, )."""
            predictions = self.predict(X)
            labels = self._validate_labels(y, len(predictions))
            return float(np.mean(predictions == labels))
            

        
            
            
            
                    
                            
                    
