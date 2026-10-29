import os, gzip, shutil, time, copy, urllib.request
import torch
import torch.nn as nn
import torchvision
#Import Numpy for arrays, matrix multiplication, and numeric calculations
import numpy as np
#Import NDArray to annotate the element types of NumPy arrays
from numpy.typing import NDArray

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
    Parameters
        -----------
        learning_rate: float, default = 0.1
            Positive step size used when updating weights and biases.
        n_epochs: int, default = 30
            Number of complete passes through the training datasets.
        batch_size: int, default = 256
            Maximum number of training samples used in one parameter update
        reg_strength: float, default = 0.0001
            Nonegative coefficient of the L2 penalty:
            0.5 * reg_strength * sum(weights ** 2).
            Biases are not regularized.
        random_state: int, default = 42
            Seed controlling the order in which training samples are shuffled.
        verbose: bool, default = True
            Where to print the training loss after each epoch
        Attributes
        -------------
        weights_: NDArray[np.float64] | None
            Weight matrix of shape (n_features, n_classes), or None before fitting
        bias_: NDArray[np.float64] | None
            Bias vector of shape (n_classes, ), or None before fitting.
        classes_: NDArray[np.int64] | None
            Class labels discovered during fitting
        loss_history_: list[float]
            Full training objective recorded after each epoch.
    """
    def __init__(
            self,
            learning_rate: float = 0.1,
            n_epochs: int = 30,
            batch_size: int = 256,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            hidden_sizes: list = (256, 128),
            activation: str = "relu",
            p_drop: float = 0.0,
            init: str = "he",
            verbose: bool = True
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
    
            #Store whether training progress should be printed
            self.verbose: bool = verbose
    
            #Weights do not exist until fit determines the input and output sizes
            self.weights_: FloatArray | None = None
    
            #Biases also remain underfined until fit is called.
            self.bias_: FloatArray | None = None
    
            #Training labels determine the available classes
            self.classes_: LabelArray | None = None
    
            #Start with an empty list of epoch losses.
            self.loss_history_: list[float] = []

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
        to initialize weight, bias, activation each layer of model
        input: sizes = [784, k_1, ..., k_{L-1}, 10]. (sizes of each layer)
        output: a model
        Linear -> Act -> (Dropout) -> ... -> Linear (output layer do not have activation).
        """
        layers = []
        L = len(sizes) - 1
        for l in range(L):
            #initialize layer l
            lin = nn.Linear(sizes[l], sizes[l + 1])

            #weight initialization
            if init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # std = sqrt(2 / k_in)
            elif init == "xavier":
                nn.init.xavier_uniform_(lin.weight)                         # U(-sqrt(6/(k_in+k_out)), +...)
            else:
                raise ValueError(init)
            #bias initialization
            nn.init.zeros_(lin.bias)

            layers.append(lin)

            if l < L - 1:
                layers.append(MLP.ACTIVATIONS[activation]())
                if p_drop > 0:
                    layers.append(nn.Dropout(p_drop))
        return nn.Sequential(*layers)

    def n_params(model):
        """
        output: the number of parameter
        """
        return sum(p.numel() for p in model.parameters())

    def forward_with_cache(
            model,
            X: FeatureArray,              #input data
            train=True      # true = use drop out, false for not
        ):
        """
        output: (a, cache)
                a = output of network
                cache[i] = (layer, a_in, a_out, extra)
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
        input:  logits (torch.tensor): pre-activate of the last layer, size (N, K),
                y: list of label (N,) 0..K-1.
        output: (loss, delta): 
                loss = mean of cross-entropy (tensor),
                delta = dJ/dlogits, size (N, K).
        """
        N = logits.shape[0]
        z = logits - logits.max(dim=1, keepdim=True).values            # subtract for max to avoid overflow
        log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))    #softmax
        loss = -log_probs[torch.arange(N), y].mean()                   # cross entropy los
        Y = torch.zeros_like(logits)
        Y[torch.arange(N), y] = 1.0                                    # set one-hot
        delta = (torch.exp(log_probs) - Y) / N                         # derivatice J with entire network output
        return loss, delta

    def linear_backward(
            layer,
            a_in: FloatArray,
            delta: FloatArray):
        """
        input:  a_in: input of current layer,
                delta: derivation E with z
        output: dW: derivation E with weight
                db: derivation E with bias
                grad_in: derivation E with a_in
        """
        dW = delta.T @ a_in
        db = delta.sum(dim=0)
        grad_in = delta @ layer.weight
        return dW, db, grad_in

    def relu_backward(a_in: FloatArray, grad: FloatArray):
        return grad * (a_in > 0).to(grad.dtype)          # f'(z) = 1[z > 0]


    def sigmoid_backward(a_out: FloatArray, grad: FloatArray):
        return grad * a_out * (1.0 - a_out)               # sigma' = a (1 - a)


    def tanh_backward(a_out: FloatArray, grad: FloatArray):
        return grad * (1.0 - a_out ** 2)                  # tanh' = 1 - tanh^2


    def dropout_backward(
            mask: bool,
            p: float,
            grad: FloatArray):
        """
        input: mask = None mean do not drop out in forward
        """
        if mask is None:
            return grad
        return grad * mask / (1.0 - p)

    def backward(
            cache: list,
            grad_out: FloatArray,
            weight_decay=0.0):
        """
        input:  grad_out = dJ/d(output of network)
        output: dJ/dX (X: input of network)."""
        grad = grad_out
        for layer, a_in, a_out, extra in reversed(cache):
            if isinstance(layer, nn.Linear):
                dW, db, grad = MLP.linear_backward(layer, a_in, grad)
                layer.weight.grad = dW + weight_decay * layer.weight   # L2 chi tren W
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
        def __init__(self, params, lr, momentum=0.0):
            self.params = list(params)
            self.lr, self.momentum = lr, momentum
            self.velocity = [torch.zeros_like(p) for p in self.params]   # Delta_w(0) = 0

        def step(self):
            with torch.no_grad():
                for p, v in zip(self.params, self.velocity):
                    if p.grad is None:
                        continue
                    v.mul_(self.momentum).sub_(self.lr * p.grad)    # Delta_w = alpha * Delta_w - gamma * g
                    p.add_(v)                                       # w = w + Delta_w

        def zero_grad(self):
            for p in self.params:
                p.grad = None

    def train_one_epoch(
            model, X, y,
            optimizer: SGDMomentum,
            batch_size=128,
            weight_decay=0.0
        ):
        """output: mean of cross-entropy loss in one epoch"""
        N = X.shape[0]
        perm = torch.randperm(N)
        total, count = 0.0, 0           #total of loss
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

    def evaluate(model, X: FeatureArray, y: LabelArray, batch_size=5000):
        """output: cross-etropy loss, accuracy"""
        total, correct = 0.0, 0
        for s in range(0, len(X), batch_size):
            logits, _ = MLP.forward_with_cache(model, X[s:s + batch_size], train=False)
            loss, _ = MLP.softmax_cross_entropy(logits, y[s:s + batch_size])
            total += loss.item() * len(logits)
            correct += (logits.argmax(dim=1) == y[s:s + batch_size]).sum().item()
        return total / len(X), correct / len(X)

    def train_model(model, epochs, lr, X_tr , y_tr, X_va, y_va,
                    momentum=0.0, batch_size=128, weight_decay=0.0,optimizer=None,
                    seed=SEED, verbose=True):
        
        torch.manual_seed(seed)
        opt = optimizer if optimizer is not None else MLP.SGDMomentum(model.parameters(), lr, momentum)
        hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}
        for ep in range(1, epochs + 1):
            t0 = time.time()
            MLP.train_one_epoch(model, X_tr, y_tr, opt, batch_size, weight_decay)
            dt = time.time() - t0
            tl, ta = MLP.evaluate(model, X_tr, y_tr)
            vl, va = MLP.evaluate(model, X_va, y_va)
            for k, v in zip(hist, [tl, ta, vl, va, dt]):
                hist[k].append(v)
            if verbose:
                print(f"epoch {ep:2d} | {dt:5.1f}s | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
                print(f"best accuracy: {max(hist["val_acc"]):.4f}")
            if not np.isfinite(tl):
                if verbose:
                    print("loss = nan/inf, early stop.")
                break
        return hist

    def fit(
            self, X: FeatureArray, y: LabelArray,
            X_val: FeatureArray, y_val: LabelArray
        ):

        #convert X into a tensor
        X = torch.as_tensor(X, dtype=torch.float32)
        #save labels's name
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        #convert label to number 0, 1, 2,... (for one-hot)
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

        #size of each layer
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
            verbose=self.verbose,
        )
        return self

    def predict(self, X: FeatureArray):
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

            y_pred = logits.argmax(dim=1).cpu().numpy()

        return self.classes_[y_pred]