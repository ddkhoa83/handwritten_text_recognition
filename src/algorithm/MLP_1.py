"""Two-hidden-layer MNIST MLP: Numpy arithmetic, manually implemented gradients.
    Architecture (default): 1x28x28 -> Flatten784 -> Dense128/ReLU -> Dense64/ReLU -> Dense10/Softmax.
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

class MPL:
    """
    Two-hidden-layer classifier with manual vanilla SGD and validation selection.

    Parameters
    -----------
    learning_rate: float, default = 0.01
        Initial vanilla SGD step size. Halved after lr_patience stale
        validation epochs. Plain gradient descent typically needs a much
        larger learning rate than Adam; 0.01 - 0.05 is a reasonable range
        for this architecture.
    n_epochs: int, default = 30
        Maximum number of complete passes over training samples
    batch_size: int, default = 64
        Maximum samples per parameter update; gradients use the actual batch size.
    reg_strength: float, default = 0.0001
        L2 coefficient: loss includes 0.5 * reg_strength * sum(W**2)
        Its gradient is added directly to the data gradient before the SGD step.
    hidden1: int, default = 128
        Width of the first fully connected hidden layer (ReLU)
    hidden2: int, default = 64
        Width of the second fully connected hidden layer (ReLU)
    image_size: int, default = 28
        Square image width; used to flatten the input (N, H*H)
    patience: int, default = 5
        Stop after this many epochs without a better validation checkpoint.
    lr_patience: int, default = 2
        Halve the step size after this many epochs without checkpoint improvement.
    random_state: int, default = 42
        Seed for initialization, shuffling, and random translations
    verbose: bool, default = True
        Print training progress.

    Attributes
    ----------
    parameters_: dict[str, NDArray[np.float64]] | None
        hidden1_weights (H*H, hidden1), hidden1_bias (hidden1, ),
        hidden2_weights (hidden1, hidden2), hidden2_bias (hidden2,),
        output_weights (hidden2, 10), output_bias (10,).
    loss_history_: list[float]
        Sample-weighted online training objectives, measured before each update
        on potentially translated images. Not a fixed-parameter training loss.
    validation_loss_history_: list[float]
        Unregularized cross-entropy on unchanged validation images after epochs
    validation_accuracy_history_: list[float]
        Validation accuracy after each epoch.
    best_epoch_ : int | None
        Selected epoch (one-based). Selection maximizes validation accuracy,
        breaking ties using lower validation cross-entropy
    
    Notes
    ------
    fit starts from scratch. With validation data it restores the best checkpoint.
    Without validation, training runs all epochs and retains the final parameters.
    Saved JSON supports inference, not exact optimizer-state training resumption
    """

    def __init__(self, learning_rate: float = 0.05, n_epochs: int = 30,
               batch_size: int = 64, reg_strength: float = 0.0001,
               hidden1: int = 128, hidden2: int = 64,
               image_size: int = 28, shift_pixels: int = 0,
               patience: int = 6, lr_patience: int = 2,
               random_state: int = 42, verbose: bool = True) -> None:
        """Validate constructor parameters and create an unfitted model."""
        for name, value in (("learning_rate", learning_rate), ("reg_strength", reg_strength)):
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"Invalid {name}.")

        for name, value in (("n_epochs", n_epochs), ("batch_size", batch_size),
                           ("hidden1", hidden1), ("hidden2", hidden2),
                           ("image_size", image_size),
                           ("patience", patience), ("lr_patience", lr_patience)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if image_size < 2:
            raise ValueError("image_size must be at least 2.")
    
        if type(shift_pixels) is not int or not 0 <= shift_pixels < image_size:
            raise ValueError("shift_pixels must be an integer in [0, image_size).")
    
        if type(random_state) is not int or random_state < 0:
            raise ValueError("random_state must be a nonnegative integers")
    
        if type(verbose) is not bool:
            raise ValueError("verbose must be a bool.")
    
        self.learning_rate = float(learning_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_strength = float(reg_strength)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.image_size = image_size
        self.patience = patience
        self.lr_patience = lr_patience
        self.random_state = random_state
        self.verbose = verbose
        self.parameters_: ParameterMap = {}
        self.classes_: LabelArray | None = None
        self.loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        self.validation_accuracy_history_: list[float] = []
        self.best_epoch_: int | None = None
        self.n_features_ = image_size * image_size

    def _initialize_parameters(self) -> None:
        """Initialize weights using He scaling and reset training history."""
        rng = np.random.default_rng(self.random_state)
        self.parameters = {}
        shapes = {
            "hidden1": (self.n_features_, self.hidden1),
            "hidden2": (self.hidden1, self.hidden2),
            "output": (self.hidden2, 10)
        }
        for name, shape in shapes.items():
            fan_in = shape[0]
            scale = np.sqrt((1.0 if name == "output" else 2.0) / fan_in)
            self.parameters_[name + "_weights"] = rng.normal(0, scale, shape)
            self.parameters_[name + "_bias"] = np.zeros(shape[1], dtype = np.float64)
        self.classes_ = np.arange(10, dtype = np.int64)
        self.loss_history_ = []
        self.validation_loss_history_ = []
        self.validation_accuracy_history_ = []
        self.best_epoch_ = None

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError when fit or load has not initialized parameters."""
        if self.parameters_ is None or self.classes_ is None:
            raise RuntimeError("Call fit or load before using this method.")

    @staticmethod
    def _log_softmax(scores: FloatArray) -> FloatArray:
        """
        Calculate stable log-probabilities for logits of shape (B, 10)

        Returns
        --------
        NDArray[np.float64]
            Shape (B, 10); exponentiating each row produces probabilities
        """
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite scores; reduce the learning rate.")

        shifted = scores - np.max(scores, axis = 1, keepdims = True)
        return shifted - np.log(np.sum(np.exp(shifted), axis = 1, keepdims = True))

    def _forward(self, X: FloatArray) -> tuple[FloatArray, ForwardCache]:
        """
        Compute logits (B, 10) and intermediated caches from images (B, 1, H, H)

        Flatted, then two ReLU hidden layers, then a linear output.
        Softmax is evaluated separately for loss of prediction.
        """

        self._check_is_fitted()
        p = self.parameters_
        flat = X.reshape(len(X), -1)
        z1 = flat @ p["hidden1_weights"] + p["hidden1_bias"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["hidden2_weights"] + p["hidden2_bias"]
        a2 = np.maximum(z2, 0)
        scores = a2 @ p["output_weights"] + p["output_bias"]
        return scores, dict(flat = flat, z1 = z1, z2 = z2, a1 = a1, a2 = a2)

    def _backward(self, score_gradient: FloatArray, cache: ForwardCache) -> ParameterMap:
        """
        Manually propagate scaled logit derivatives (B, 10) through all layers.

        Return parameter-shaped data gradients. The caller adds regularization.
        All weights are unchanged until the complete backwawrd pass has finished
        """
        p, c = self.parameters_, cache
        gradients = {}
        # Output layer: dW = a2^T . dS, db = sum(dS), dA2 = dS . W^T
        gradients["output_weights"] = c["a2"].T @ score_gradient
        gradients["output_bias"] = score_gradient.sum(axis=0)
        a2_gradient = score_gradient @ p["output_weights"].T
        z2_gradient = a2_gradient * (c["z2"] > 0)
        # Second hidden layer: dW = a1^T . dZ2, db = sum(dZ2), dA1 = dZ2 . W^T
        gradients["hidden2_weights"] = c["a1"].T @ z2_gradient
        gradients["hidden2_bias"] = z2_gradient.sum(axis=0)
        a1_gradient = z2_gradient @ p["hidden2_weights"].T
        z1_gradient = a1_gradient * (c["z1"] > 0)
        # First hidden layer: dW = flat^T . dZ1, db = sum(dZ1)
        gradients["hidden1_weights"] = c["flat"].T @ z1_gradient
        gradients["hidden1_bias"] = z1_gradient.sum(axis=0)
        return gradients

    def _regularization_loss(self) -> float:
        """Return 0.5 * reg_strength * sum of squared weights, excluding biases."""
        assert self.parameters_ is not None
        return 0.5 * self.reg_strength * sum(
            float(np.sum(self.parameters_[name] ** 2))
            for name in self.parameters_ if name.endswith("weights")
        )
        
    def _loss_and_gradients(self, X: FloatArray, y: LabelArray) -> tuple[float, ParameterMap]:
        """
        Compute the regularized mini-batch loss and manually derived gradients.

        Parameters
        ------------
        X : NDArray[np.float64]
            Shape (B, 1, H, H), validated normalized images
        y : NDArray[np.int64]
            Shape (B, ), correct digit indices.

        Returns
        -----------
        loss : float
            Mean cross-entropy plus the L2 weight penalty
        gradients : dict[str, NDArray[np.float64]]
            Gradients of the same objective, matching all parameter shapes.
        """
        scores, cache = self._forward(X)
        log_probabilities = self._log_softmax(scores)
        count = len(X)
        data_loss = -float(np.mean(log_probabilities[np.arange(count), y]))
        #Analytic derivative of softmax + cross-entropy; no derivative library.
        errors = np.exp(log_probabilities)
        errors[np.arange(count), y] -= 1
        errors /= count
        gradients = self._backward(errors, cache)

        assert self.parameters_ is not None
        for name in (key for key in self.parameters_ if key.endswith("weights")):
            gradients[name] += self.reg_strength * self.parameters_[name]
        return data_loss + self._regularization_loss(), gradients

    def _sgd_update(self, gradients: ParameterMap, rate: float) -> None:
        """Update parameters with vanilla gradient descent: theta <- theta - rate*grad"""
        assert self.parameters_ is not None
        for name, gradient in gradients.items():
            self.parameters_[name] -= rate*gradient
            if not np.isfinite(self.parameters_[name]).all():
                raise FloatingPointError("Training diverged; lower learning_rate.")

    def _evaluate(self, X: np.ndarray, y: LabelArray) -> tuple[float, float]:
        """Return mean data cross-entropy and accuracy, using unchanged parameters"""
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), self.batch_size):
            batch = np.asarray(X[start:start + self.batch_size], dtype = np.float64)
            labels = y[start: start + self.batch_size]
            scores, _ = self._forward(batch)
            logp = self._log_softmax(scores)
            total_loss -= float(logp[np.arange(len(batch)), labels].sum())
            correct += int(np.sum(logp.argmax(axis = 1) == labels))

        return total_loss / len(X), correct / len(X)

    def fit(self, X: FeatureArray | FloatArray, y: LabelArray,
           X_val: FeatureArray | FloatArray | None = None,
           y_val: LabelArray | None = None) -> "MLP":
        """Reset and train with vanilla SGD; optionally restore the best checkpoint
        
        Parameters
        ----------
        X, X_val : array-like
            Validation images must be disjoint from training and test images.
        y, y_val : integer array-like
            Corresponding digit labels of shape (N,)

        Returns
        ---------
        MLP
            The trained model. With validation, parameters correspond to best_epoch_.
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val, or neither.")

        self._initialize_parameters()
        rng = np.random.default_rng(self.random_state)

        stale, lr_stale = 0, 0
        rate = self.learning_rate
        best_key, best_parameters = (-1.0, -float("inf")), None

        for epoch in range(self.n_epochs):
            started = time.perf_counter()
            order = rng.permutation(len(X))
            total_loss = 0.0
            for start in range(0, len(X), self.batch_size):
                indices = order[start: start + self.batch_size]
                batch = X[indices]
                loss, gradients = self._loss_and_gradients(batch, y[indices])
                if not math.isfinite(loss):
                    raise FloatingPoinError("Nonfinite loss; lower learning_rate.")
                total_loss += loss*len(indices)
                self._sgd_update(gradients, rate)
            self.loss_history_.append(total_loss / len(X))
            message = (f"Epoch {epoch + 1:003} / {self.n_epochs:03d} | "
                      f"Train objective (online): {self.loss_history_[-1]:.6f} | LR: {rate:.6g}")

            if X_val is not None:
                val_loss, val_accuracy = self._evaluate(X_val, y_val)
                self.validation_loss_history_.append(val_loss)
                self.validation_accuracy_history_.append(val_accuracy)
                message += f" | Val loss: {val_loss:.6f} | Val accuracy: {val_accuracy:.4%}"
                key = (val_accuracy, -val_loss)
                if key > best_key:
                    best_key = key
                    best_parameters = {name: value.copy() for name, value in self.parameters_.items()}
                    self.best_epoch_ = epoch + 1
                    stale = lr_stale = 0
                else:
                    stale += 1
                    lr_stale += 1
                    if lr_stale >= self.lr_patience:
                        rate = max(rate * 0.5, self.learning_rate * 0.01)
                        lr_stale = 0

            if self.verbose:
                print(message + f" | {time.perf_counter() - started: .1f}s", flush = True)
            if X_val is not None and stale >= self.patience:
                if self.verbose:
                    print("Early stopping: validation checkpoint did not improve.", flush=True)
                break
        if best_parameters is not None:
            self.parameters_ = best_parameters

            if self.verbose:
                print(f"Restored epoch {self.best_epoch_}; validation accuracy: {best_key[0]:.4%}", flush=True)
        else:
            self.best_epoch_ = len(self.loss_history_)
        return self

    def predict_proba(self, X: FeatureArray | FloatArray) -> FloatArray:
        """
        Return probabilities of shape (N, 10) for normalized images

        Parameters
        -----------
        X : array-like
            Shape (N, 1, H, H)
        Returns
        --------
        NDArray[np.float64]
            Probabilities; rows sum approximately to one.
        """
        result = np.empty((len(X), 10), dtype = np.float64)
        for start in range(0, len(X), self.batch_size):
            batch = np.asarray(X[start:start + self.batch_size], dtype = np.float64)
            scores, _ = self._forward(batch)
            result[start: start + len(batch)] = np.exp(self._log_softmax(scores))
        return result

    def predict(self, X: FeatureArray | FloatArray) -> LabelArray:
        """Return digit labels (N, )"""
        return np.argmax(self.predict_proba(X), axis = 1).astype(np.int64)

    def score(self, X: FeatureArray | FloatArray, y: LabelArray) -> float:
        """Return classification accuracy in [0, 1] for X and its labels (N, )."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
        
        
                
            
                
        
    
