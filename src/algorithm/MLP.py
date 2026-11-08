import os, gzip, shutil, time, copy, urllib.request
import torch
import torch.nn as nn
import torchvision
#Import Numpy for arrays, matrix multiplication, and numeric calculations
import numpy as np
#Import NDArray to annotate the element types of NumPy arrays
from numpy.typing import NDArray

import matplotlib.pyplot as plt


#Input features produced by the preprocessing code use float32
FeatureArray = NDArray[np.float32]

#Model calculations use float64 for additional numeric precision
FloatArray = NDArray[np.float64]

#Class labels are stored as signed 64-bit integers.
LabelArray = NDArray[np.int64]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class MLP:
    """
    Multilayer perceptron classifier implemented with PyTorch modules.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Gradient-descent step size; must be finite and positive.
    n_epochs : int, default=20
        Maximum number of training epochs.
    batch_size : int, default=256
        Number of samples processed in each mini-batch.
    momentum : float, default=0.0
        Momentum coefficient used by the SGD optimizer.
    weight_decay : float, default=0.0
        L2 regularization strength applied to weights, not biases.
    hidden_sizes : list, default=[256, 128]
        Number of neurons in each hidden layer.
    activation : str, default="relu"
        Hidden-layer activation: "relu", "sigmoid", or "tanh".
    p_drop : float, default=0.0
        Dropout probability applied after hidden-layer activations.
    init : str, default="he"
        Weight initialization method: "he", "xavier", or "const".
    early_stop : bool, default=False
        Whether to stop training when validation loss stops improving.
    patience : int, default=0
        Number of consecutive epochs without improvement before stopping.
    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    model_ : nn.Sequential or None
        Trained neural network; None before fitting.
    weights_ : FloatArray or None
        Reserved weight attribute; not populated by the current implementation.
    bias_ : FloatArray or None
        Reserved bias attribute; not populated by the current implementation.
    classes_ : LabelArray or None
        Original class labels, stored in sorted order.
    history_ : dict
        Training and validation metrics recorded for each epoch.
    """
    def __init__(
            self,
            learning_rate: float = 0.1,
            n_epochs: int = 20,
            batch_size: int = 256,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            hidden_sizes: list = [256, 128],
            activation: str = "relu",
            p_drop: float = 0.0,
            init: str = "he",
            early_stop: bool = False,
            patience: int = 0,
            verbose: bool = False
        ) -> None:
            """
            Validate configuration values and initialize model attributes.
    
            Parameters
            -----------
            learning_rate: float
                Positive gradient-descent step size.
            n_epochs: int
                Positive number of training epochs
            batch_size: int
                Positive maximum number of samples per mini-batch
            verbose: bool
                Whether to display training progress
    
            Returns
            -----------
            None
                Initializes the model without training it.
            """
            #Reject a nonfinite, zero, or negative learning rate.
            if not np.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("learning_rate must be finite and positive.")
    
            #Require a positive integer number of epochs.
            if not isinstance(n_epochs, int) or n_epochs <= 0:
                raise ValueError("n_epochs must be a positive integer.")
    
            #Require a positive integer number of batch size.
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")

            #Require a positive weight decay.
            if not np.isfinite(weight_decay) or weight_decay < 0:
                raise ValueError("weight_decay must be finite and nonnegative.")

            #Require a positive integer number of batch size.
            if not np.isfinite(momentum) or momentum < 0:
                raise ValueError("momentum must be a positive.")

            #Store the learning rate as a Python float.
            self.learning_rate: float = float(learning_rate)
    
            #Store number of passes through the training data.
            self.n_epochs: int = n_epochs
    
            #Store the maximum number of samples per update.
            self.batch_size: int = batch_size

            self.weight_decay = weight_decay

            self.momentum = momentum

            self.hidden_sizes = tuple(hidden_sizes)

            self.activation = activation

            self.p_drop = p_drop

            self.init = init

            self.early_stop = early_stop

            self.patience = patience
    
            #Store whether training progress should be printed
            self.verbose: bool = verbose

            self.model_: nn.Sequential | None = None
    
            #Weights do not exist until fit determines the input and output sizes
            self.weights_: FloatArray | None = None
    
            #Biases also remain underfined until fit is called.
            self.bias_: FloatArray | None = None
    
            #Training labels determine the available classes
            self.classes_: LabelArray | None = None
    
            #Start with an empty list of epoch losses.
            self.history_: dict[float] = []

    ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    def build_mlp(
            self,
            sizes: list ,
            activation="relu",
            p_drop=0.0,
            init="he",
            const=0.01,
            seed=SEED
        ):
        """
        Build the network from its layer sizes.

        Parameters
        ----------
        sizes : list
            Neuron counts, including input, hidden, and output layers.
        activation : str
            Hidden-layer activation function.
        p_drop : float
            Dropout probability after each hidden activation.
        init : str
            Weight initialization: "he", "xavier", or "const".
        const : float
            Constant weight value when init="const".
        seed : int
            Seed argument reserved for initialization control.

        Returns
        -------
        nn.Sequential
            Linear and hidden activation/dropout layers. The output layer
            returns logits and has no Softmax activation.
        """
        layers = []
        L = len(sizes) - 1
        for l in range(L):
            # Create linear layer l.
            lin = nn.Linear(sizes[l], sizes[l + 1])

            # Initialize weights.
            if init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # std = sqrt(2 / k_in)
            elif init == "xavier":
                nn.init.xavier_uniform_(lin.weight)                         # U(-sqrt(6/(k_in+k_out)), +...)
            elif init == "const":
                nn.init.constant_(lin.weight, const)
            else:
                raise ValueError(init)
            # Initialize biases.
            nn.init.zeros_(lin.bias)

            layers.append(lin)

            if l < L - 1:
                layers.append(MLP.ACTIVATIONS[activation]())
                if p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
        return nn.Sequential(*layers)

    def forward_with_cache(
            model,
            X: FeatureArray,              #input data
            train=True      # True enables dropout; False disables it.
        ):
        """
        Run a forward pass and save intermediate values for backpropagation.

        Parameters
        ----------
        model : nn.Sequential
            Network to evaluate.
        X : FeatureArray
            Input batch with shape (n_samples, n_features).
        train : bool, default=True
            Whether dropout is active.

        Returns
        -------
        tuple
            Network output (logits) and a cache containing each layer's
            inputs, outputs, and optional dropout mask.
        """
        cache = []
        a = X
        for layer in model:
            a_in = a
            extra = None
            if isinstance(layer, nn.Dropout):
                if train and layer.p > 0:
                    extra = (torch.rand_like(a_in) >= layer.p).to(a_in.dtype)   # M ~ Bernoulli(1 - p)
                    a = a_in * extra / (1.0 - layer.p)
                else:
                    a = a_in
            else:
                a = layer(a_in)
            cache.append((layer, a_in, a, extra))
        return a, cache

    def softmax_cross_entropy(
            logits: torch.Tensor,
            y: LabelArray):
        """
        Compute mean cross-entropy loss and its gradient with respect to logits.

        Parameters
        ----------
        logits : torch.Tensor
            Unnormalized class scores with shape (n_samples, n_classes).
        y : LabelArray or torch.Tensor
            Encoded class indices in the range [0, n_classes - 1].

        Returns
        -------
        tuple
            Scalar mean loss and the gradient d(loss)/d(logits).
        """
        N = logits.shape[0]
        z = logits - logits.max(dim=1, keepdim=True).values            # Subtract the row maximum for numerical stability.
        log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))    # Compute log-softmax values.
        loss = -log_probs[torch.arange(N), y].mean()                   # Select the log-probability of the correct class.
        Y = torch.zeros_like(logits)
        Y[torch.arange(N), y] = 1.0                                    # Create the one-hot encoded target matrix.
        delta = (torch.exp(log_probs) - Y) / N                         # Gradient of the loss with respect to logits.
        return loss, delta

    def linear_backward(
            layer,
            a_in: FloatArray,
            delta: FloatArray):
        """
        Compute gradients for a linear layer.

        Parameters
        ----------
        layer : nn.Linear
            Linear layer whose gradients are calculated.
        a_in : torch.Tensor
            Input to the layer.
        delta : torch.Tensor
            Gradient with respect to the layer output before activation.

        Returns
        -------
        tuple
            Gradients for weights, biases, and the layer input.
        """
        dW = delta.T @ a_in
        db = delta.sum(dim=0)
        grad_in = delta @ layer.weight
        return dW, db, grad_in

    def relu_backward(a_in: FloatArray, grad: FloatArray):
        """Backpropagate through ReLU using its input and upstream gradient."""
        return grad * (a_in > 0).to(grad.dtype)          # f'(z) = 1[z > 0]


    def sigmoid_backward(a_out: FloatArray, grad: FloatArray):
        """Backpropagate through Sigmoid using its output and upstream gradient."""
        return grad * a_out * (1.0 - a_out)               # sigma' = a (1 - a)


    def tanh_backward(a_out: FloatArray, grad: FloatArray):
        """Backpropagate through Tanh using its output and upstream gradient."""
        return grad * (1.0 - a_out ** 2)                  # tanh' = 1 - tanh^2


    def dropout_backward(
            mask: bool,
            p: float,
            grad: FloatArray):
        """
        Backpropagate through dropout.

        Parameters
        ----------
        mask : torch.Tensor or None
            Dropout mask from the forward pass; None means dropout was inactive.
        p : float
            Dropout probability.
        grad : torch.Tensor
            Gradient from the next layer.

        Returns
        -------
        torch.Tensor
            Gradient with respect to the dropout input.
        """
        if mask is None:
            return grad
        return grad * mask / (1.0 - p)

    def backward(
            cache: list,
            grad_out: FloatArray,
            weight_decay=0.0):
        """
        Backpropagate gradients through all cached layers.

        Parameters
        ----------
        cache : list
            Intermediate values returned by forward_with_cache().
        grad_out : torch.Tensor
            Gradient with respect to the network output.
        weight_decay : float, default=0.0
            L2 regularization strength for weight gradients.

        Returns
        -------
        torch.Tensor
            Gradient with respect to the network input.
        """
        grad = grad_out
        for layer, a_in, a_out, extra in reversed(cache):
            if isinstance(layer, nn.Linear):
                dW, db, grad = MLP.linear_backward(layer, a_in, grad)
                layer.weight.grad = dW + weight_decay * layer.weight   # Apply L2 regularization to weights only.
                layer.bias.grad = db
            elif isinstance(layer, nn.ReLU):
                grad = MLP.relu_backward(a_in, grad)
            elif isinstance(layer, nn.Sigmoid):
                grad = MLP.sigmoid_backward(a_out, grad)
            elif isinstance(layer, nn.Tanh):
                grad = MLP.tanh_backward(a_out, grad)
            elif isinstance(layer, nn.Dropout):
                grad = MLP.dropout_backward(extra, layer.p, grad)
            else:
                raise TypeError(f"This module is not supported {type(layer).__name__}")
        return grad

    class SGDMomentum:
        """Mini-batch stochastic gradient descent optimizer with momentum.

        Parameters
        ----------
        params : iterable
            Model parameters to update.
        lr : float
            Learning rate.
        momentum : float, default=0.0
            Momentum coefficient.

        Attributes
        ----------
        params : list
            Parameters updated by the optimizer.
        velocity : list
            Momentum buffers corresponding to the parameters.
        """
        def __init__(self, params, lr, momentum=0.0):
            self.params = list(params)
            self.lr, self.momentum = lr, momentum
            self.velocity = [torch.zeros_like(p) for p in self.params]   # Delta_w(0) = 0

        def step(self):
            """Update parameters using their gradients and momentum buffers."""
            with torch.no_grad():
                for p, v in zip(self.params, self.velocity):
                    if p.grad is None:
                        continue
                    v.mul_(self.momentum).sub_(self.lr * p.grad)    # Delta_w = alpha * Delta_w - gamma * g
                    p.add_(v)                                       # w = w + Delta_w

        def zero_grad(self):
            """Clear stored parameter gradients before the next backward pass."""
            for p in self.params:
                p.grad = None

    def train_one_epoch(
            model, X, y,
            optimizer: SGDMomentum,
            batch_size=256,
            weight_decay=0.0
        ):
        """
        Train the network for one epoch using mini-batch SGD.

        Parameters
        ----------
        model : nn.Sequential
            Network being trained.
        X : torch.Tensor
            Training features.
        y : torch.Tensor
            Encoded training labels.
        optimizer : SGDMomentum
            Optimizer used for parameter updates.
        batch_size : int
            Number of samples per mini-batch.
        weight_decay : float
            L2 regularization strength.

        Returns
        -------
        float
            Mean cross-entropy loss over the epoch.
        """
        N = X.shape[0]
        perm = torch.randperm(N)
        total, count = 0.0, 0           # Accumulate the loss across mini-batches.
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            logits, cache = MLP.forward_with_cache(model, X[idx], train=True)
            optimizer.zero_grad()
            loss, delta = MLP.softmax_cross_entropy(logits, y[idx])
            MLP.backward(cache, delta, weight_decay)
            optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)
        return total / count

    def evaluate(model, X: FeatureArray, y: LabelArray, batch_size=256):
        """
        Evaluate the network on a dataset.

        Parameters
        ----------
        model : nn.Sequential
            Network to evaluate.
        X : FeatureArray or torch.Tensor
            Evaluation features.
        y : LabelArray or torch.Tensor
            Encoded evaluation labels.
        batch_size : int
            Number of samples processed per batch.

        Returns
        -------
        tuple
            Mean cross-entropy loss and classification accuracy.
        """
        total, correct = 0.0, 0
        for s in range(0, len(X), batch_size):
            logits, _ = MLP.forward_with_cache(model, X[s:s + batch_size], train=False)
            loss, _ = MLP.softmax_cross_entropy(logits, y[s:s + batch_size])
            total += loss.item() * len(logits)
            correct += (logits.argmax(dim=1) == y[s:s + batch_size]).sum().item()
        return total / len(X), correct / len(X)

    def train_model(model, epochs, lr, X_tr , y_tr, X_va, y_va,
                    momentum=0.0, batch_size=128, weight_decay=0.0, optimizer=None,
                    early_stop= False, patience = 0, seed=SEED, verbose=True):
        """
        Train a network and record training/validation metrics.

        Parameters
        ----------
        model : nn.Sequential
            Network to train.
        epochs : int
            Maximum number of training epochs.
        lr : float
            Learning rate.
        X_tr, y_tr : torch.Tensor
            Training features and encoded labels.
        X_va, y_va : torch.Tensor
            Validation features and encoded labels.
        momentum : float
            SGD momentum coefficient.
        batch_size : int
            Mini-batch size.
        weight_decay : float
            L2 regularization strength.
        optimizer : SGDMomentum or None
            Optional preconfigured optimizer.
        early_stop : bool
            Whether to stop when validation loss stops improving.
        patience : int
            Number of epochs to wait for improvement.
        seed : int
            Random seed for training.
        verbose : bool
            Whether to print epoch metrics.

        Returns
        -------
        dict
            Lists of train/validation losses, accuracies, and epoch times.
            With early stopping, also includes "best_epoch".
        """

        torch.manual_seed(seed)

        opt = optimizer if optimizer is not None else MLP.SGDMomentum(model.parameters(), lr, momentum)

        hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}

        if early_stop and patience > 0: early_stop = True
        if early_stop:
            best_loss = float("inf")
            best_epoch = 0
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0


        for ep in range(1, epochs + 1):
            t0 = time.time()
            MLP.train_one_epoch(model, X_tr, y_tr, opt, batch_size, weight_decay)
            dt = time.time() - t0
            train_loss, train_acc = MLP.evaluate(model, X_tr, y_tr)
            val_loss, val_acc = MLP.evaluate(model, X_va, y_va)

            for k, v in zip(hist, [train_loss, train_acc, val_loss, val_acc, dt]):
                hist[k].append(v)
            if verbose:
                print(f"epoch {ep:2d} | {dt:5.1f}s | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
            if not np.isfinite(train_loss):
                if verbose:
                    print("Loss is NaN or infinite; stopping training.")
                break

            #early stopping
            if early_stop:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = ep
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                    if verbose:
                        print("Loss is NaN or infinite; stopping training.")
                    break

                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep}; best epoch = {best_epoch}.")
                    break

        if early_stop:
            model.load_state_dict(best_state)
            hist["best_epoch"] = best_epoch
            return hist

        return hist

    def fit(
            self, X: FeatureArray, y: LabelArray,
            X_val: FeatureArray, y_val: LabelArray
        ):
        """
        Build and train the classifier.

        Parameters
        ----------
        X, y : array-like
            Training features and original class labels.
        X_val, y_val : array-like
            Validation features and original class labels.

        Returns
        -------
        self
            The fitted classifier.
        """
        # Start timing the training process
        start_time = time.time()  

        # Start timing the training process
        start_time = time.time()  

        # Convert features to a PyTorch tensor.
        X = torch.as_tensor(X, dtype=torch.float32)
        # Store the original class labels.
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        # Encode class labels as integer indices.
        y_encode = np.searchsorted(self.classes_, y)
        y = torch.as_tensor(y_encode, dtype=torch.long)

        if X_val is None:
            X_val = X
            y_val = y
        else:
            X_val = torch.as_tensor(X_val, dtype=torch.float32)
            y_val = np.asarray(y_val)
            y_val = np.searchsorted(self.classes_, y_val)
            y_val = torch.as_tensor(y_val, dtype=torch.long)

        # Determine the number of neurons in each layer.
        sizes = [X.shape[1], *self.hidden_sizes, len(self.classes_)]

        self.model_ = self.build_mlp(
            sizes,
            activation= self.activation,
            p_drop= self.p_drop,
            init=self.init
        )
        self.history_ = MLP.train_model(
            self.model_,
            epochs=self.n_epochs,
            lr=self.learning_rate,
            momentum=self.momentum,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            X_tr=X,
            y_tr=y,
            X_va=X_val,
            y_va=y_val,
            early_stop=self.early_stop,
            patience=self.patience,
            verbose=self.verbose,
        )
        end_time = time.time()
        self.count_training_time = end_time - start_time
        return self

    def predict(self, X: FeatureArray):
        """
        Predict the original class label for each input sample.

        Parameters
        ----------
        X : FeatureArray
            Input samples with shape (n_samples, n_features).

        Returns
        -------
        LabelArray
            Predicted labels in the original label format.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = torch.as_tensor(X, dtype=torch.float32)

        # Gradients are unnecessary during inference.
        with torch.no_grad():
            logits, _ = MLP.forward_with_cache(
                self.model_,
                X,
                train=False
            )

            y_pred = logits.argmax(dim=1).cpu().numpy()

        return self.classes_[y_pred]
    def predict_proba(self, X: FeatureArray):
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = torch.as_tensor(X, dtype=torch.float32)

        # no need to compute gradients when predict
        with torch.no_grad():
            logits, _ = MLP.forward_with_cache(
                self.model_,
                X,
                train=False
            )

            z = logits - logits.max(dim=1, keepdim=True).values
            probs = torch.exp(z) / torch.exp(z).sum(dim=1, keepdim=True)
            return probs.cpu().numpy()

    def predict_prob(self, X: FeatureArray):
        """
        Return the predicted probability of each class for every sample.

        Parameters
        ----------
        X : FeatureArray
            Input samples with shape (n_samples, n_features).

        Returns
        -------
        NDArray
            Class probabilities as percentages (0 to 100). Column order
            follows self.classes_.
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = torch.as_tensor(X, dtype=torch.float32)

        with torch.no_grad():
            logits, _ = MLP.forward_with_cache(
                self.model_,
                X,
                train=False
            )

            prob = torch.softmax(logits, dim=1).cpu().numpy()

        return prob * 100

    @staticmethod
    def layer_grad_norms(model, X, y):
        """
        Calculate the weight-gradient norm for each linear layer.

        Parameters
        ----------
        model : nn.Sequential
            Network to inspect.
        X : FeatureArray
            Input samples.
        y : LabelArray
            Encoded class indices.

        Returns
        -------
        list
            Frobenius norms of weight gradients, from first to last layer.
        """
        """Return ||dJ/dW^(l)||_F for linear layers, from first to last."""
        # Convert features to a PyTorch tensor.
        X = torch.as_tensor(X, dtype=torch.float32)
        # Store the original class labels.
        y = torch.as_tensor(y, dtype=torch.long)

        logits, cache = MLP.forward_with_cache(model, X, train=True)
        loss, delta = MLP.softmax_cross_entropy(logits, y)
        MLP.backward(cache, delta, weight_decay=0.0)
        return [
            layer.weight.grad.norm().item()
            for layer in model if isinstance(layer, nn.Linear)
        ]


    @staticmethod
    def acc_loss_history_plot(histories, title=""):
        """
        Plot training and validation loss and accuracy across epochs.

        Parameters
        ----------
        histories : dict
            Mapping from model names to their training-history dictionaries.
        title : str, default=""
            Figure title.

        Returns
        -------
        None
            Displays the figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        for name, h in histories.items():
            ep = np.arange(1, len(h["val_loss"]) + 1)
            line, = axes[0].plot(ep, h["train_loss"], "-", label=f"{name} (train)")
            axes[0].plot(ep, h["val_loss"], "--", color=line.get_color(), label=f"{name} (val)")
            axes[1].plot(ep, h["train_acc"], "-", color=line.get_color())
            axes[1].plot(ep, h["val_acc"], "--", color=line.get_color())

        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
        axes[0].legend(fontsize=8)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap(M, row_labels, col_labels, row_name, col_name, title, cell_note=None):
        """
        Display a matrix as an annotated heatmap.

        Parameters
        ----------
        M : NDArray
            Matrix of values to display.
        row_labels, col_labels : list
            Labels for matrix rows and columns.
        row_name, col_name : str
            Axis labels.
        title : str
            Figure title.
        cell_note : callable or None
            Optional function returning extra text for each cell.

        Returns
        -------
        None
            Displays the heatmap.
        """
        fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2, 0.9 * len(row_labels) + 1.5))
        ax.imshow(np.nan_to_num(M, nan=np.nanmin(M) if np.isfinite(M).any() else 0), cmap="viridis")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                txt = "diverged" if np.isnan(M[i, j]) else f"{M[i, j]:.4f}"
                if cell_note is not None:
                    txt += "\n" + cell_note(i, j)
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if np.isnan(M[i, j]) or M[i, j] < np.nanmedian(M) else "black")
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
        ax.set_xlabel(col_name); ax.set_ylabel(row_name); ax.set_title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def relu_output_stds(model, X):
        """
        Measure the output standard deviation of each ReLU layer.

        Parameters
        ----------
        model : nn.Sequential
            Network to inspect.
        X : FeatureArray
            Input samples.

        Returns
        -------
        list
            Standard deviation of each ReLU layer's output.
        """
        # Convert features to a PyTorch tensor.
        X = torch.as_tensor(X, dtype=torch.float32)

        _, cache = MLP.forward_with_cache(model, X, train=False)
        return [a_out.std().item() for layer, _, a_out, _ in cache if isinstance(layer, nn.ReLU)]


    @staticmethod
    def n_params(model):
        """
        Count the trainable and non-trainable parameters in a model.

        Parameters
        ----------
        model : nn.Module
            PyTorch model.

        Returns
        -------
        int
            Total number of scalar parameters.
        """
        """
        output: the number of parameter
        """
        return sum(p.numel() for p in model.parameters())
