#Allow annotations referring to this class
from __future__ import annotations

#Parse the local datasets path and training settings.
import argparse

#Represent filesystem paths independently of OS
from pathlib import Path

#Declare readable array type aliases.
from typing import TypeAlias

#Supply array operations; implement all learning ourselves
import numpy as np
#Describe input and output array types
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64] #Dense real-valued matrix/vector.
IntArray: TypeAlias = NDArray[np.int64] #Integer class labels or encoded indices.
IndexArray: TypeAlias = NDArray[np.intp] #Native-size array indexing integers

def validate_features(features: ArrayLike) -> FloatArray:
    """
    Convert finite feature data to a float64 matrix.

    Parameters
    ----------
    feature: ArrayLike, shape (n_samples, n_features)
        Numeric feature values; preprocessing is the scaller's responsibility
    Returns
    ---------
    FloatArray, shape (n_samples, n_features)
        Validated data, possibly sharing memory with the input

    Raises
    --------
    ValueError
        If either dimension is empty, input is not 2-D, or values are nonfinite
    """
    #Standardize numerical precision
    matrix: FloatArray = np.asarray(features, dtype = np.float64)
    #Require actual samples and features
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("Features must be a nonempty 2-D matrix.")
    #Prevent NaN/Inf from containating gradients.
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values.")
    return matrix

def validate_labels(labels: ArrayLike, n_samples: int) -> IntArray:
    """
    Validate integer labels without silently truncating fractional values

    Parameters
    ----------
    labels : ArrayLike, shape (n_samples,)
        Integer class IDs; the SVM supports nonconsecutive IDs too.
    n_sampls : int
        Required number of labels, matching the feature matrix rows.

    Returns
    --------
    IntArray, shape(n_samples,)
        Valid integer labels.

    Raises
    -------
    ValueError
        If shape, integer dtype, or conversion to int64 is invalid
    """
    #Inspect the original dtype before conversion
    values: NDArray = np.asarray(labels)
    if values.shape != (n_samples,) or values.dtype.kind not in "iu": #Disallow column vectors/floats
        raise ValueError("labels must be a 1-D integer array matching the samples.")
    #Guard overflow
    if values.size and (int(values.min()) < -(2**63) or int(values.max()) >= 2**63):
        raise ValueError("labels must fit into int64.")
    return values.astype(np.int64, copy = False) #Preserve labels in a consistent integer dtype

class MulticlassLinearSVM:
    """
    Learn all class score functions jointly with mini-batch subgradient descent.

    Attributes
    -----------
    weights_ : FloatArray or None, shape (n_features, n_classes)
        One learned weight column per class, None before fit.
    bias_ : FloatArray or None, shape (n_classes,)
        Unregularized intercepts; None before fit.
    classes_ : IntArray or None, shape (n_classes,)
        Sorted original labels corresponding to score columns.
    loss_history_ : list[float]
        Sample-weighted mean of batch objectives evaluated BEFORE each update,
        one value per epoch. This is not the final epoch model's full data loss.
    """
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 30,
                batch_size: int = 256, reg_strength: float = 0.0001,
                learning_rate_decay: float = 0.05, random_state: int = 42,
                verbose: bool = True) -> None:
        """Initialize settings without training.

        Parameters
        -----------
        learning_rate: float, positive finite
            Initial step size for weights and bias.
        n_epochs: int, positive
            Number of complete passes over the training set.
        batch_size: int, positive
            Maximum samples half the squared Frobenius norm of weights.
        learning_rate_decay: float, nonnegative finite
            Epoch schedule: learning_rate / (1 + decay * zero_based_epoch).
        random_state: int, nonnegative
            Seed for initialization and shuffling; fit resets the generator.
        verbose : bool
            Whether to print traning progress

        Returns
        ----------
        None
            Stores configuration and initializes unfitted attributes
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0: #Check the step size.
            raise ValueError("learning_rate must be positive and finite.") 
        if not np.isfinite(reg_strength) or reg_strength < 0: #Check regularization strength.
            raise ValueError("reg_strength must be nonnegative and finite.")
        if not np.isfinite(learning_rate_decay) or learning_rate_decay < 0: #Check the schedule.
            raise ValueError("learning_rate_decay must be nonnegative and finite.")
        if type(n_epochs) is not int or n_epochs < 1: #Require an actual positive epoch count.
            raise ValueError("n_epochs must be a positive integer.")
        if type(batch_size) is not int or batch_size < 1: #Require valid bathc slicing.
            raise ValueError("batch_size must be a positive integer.")
        if type(random_state) is not int or random_state < 0: #Validate the seed
            raise ValueError("random_state must be a nonnegative integer.")


        self.learning_rate: float = float(learning_rate)
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.reg_strength: float = float(reg_strength)
        self.learning_rate_decay: float = float(learning_rate_decay) #Store epoch decay.
        self.random_state: int = random_state #Store the reproducibility seed.
        self.verbose: bool = verbose
        self.weights_: FloatArray | None = None
        self.bias_: FloatArray | None = None
        self.classes_: IntArray | None = None
        self.loss_history_: list[float] = []

    def _require_fiited(self) -> tuple[FloatArray, FloatArray, IntArray]:
        """
        Return initialized weights (D, K), bias (K, ) and classes (K,).

        Takes no arguments besides self. Raises RuntimeError before parameters
        exist; returns references, not copies. Internal fit uses this after
        parameter initialization, before completing optimization
        """
        if self.weights_ is None or self.bias_ is None or self.classes_ is None: #Check all state.
            raise RuntimeError("Call fit before using the model.")

        return self.weights_, self.bias_, self.classes_

    def _loss_and_gradients(self, features: FloatArray,
                           encoded_labels: IntArray) -> tuple[float, FloatArray, FloatArray]:
        """Compute a batch objective and its subgradients
        
        Parameters
        -----------
        features: FloatArray, shape(B, D)
            Validated nonempty mini-batch, B samples and D features.
        encoded_labels: IntArray, shape (B, )
            Internal class indices in [0, K); not arbitrary orginal labels.

        Returns
        ----------
        loss: float
            Mean sum of incorrect-class hinge violations plus weight penalty
        weight_gradient: FloatArray, shape (D, K)
            X.T @ score_gradient / B + reg_strength * weights
        bias_gradient: FloatArray, shape (K, )
            Sum of score gradients across sample divided by B.

        Notes
        ------
        Internal method: assumes validated shapes and initialized parameters.
        At an exact zero margin, it chooses the valid zero hinge subgradient.
        """
        
        #Declare each returned state type
        weights: FloatArray; bias: FloatArray; classes: IntArray
        #Obtain current model parameters.
        weights, bias, classes = self._require_fiited()
        #Use actual B, including the last short batch
        batch_count: int = features.shape[0]
        #Compute (B, K) class scores via broadcasting
        scores: FloatArray = features @ weights + bias
        #Index each sample's correct score
        row_indices: IndexArray = np.arange(batch_count)
        #Gather shape (B, 1)
        correct_scores: FloatArray = scores[row_indices, encoded_labels, None]
        #Apply hinge penalties.
        margins: FloatArray = np.maximum(0.0, 1 + scores - correct_scores)
        margins[row_indices, encoded_labels] = 0.0
        #Average per SAMPLE, not per class.
        data_loss: float = float(margins.sum() / batch_count)
        #Penalize W only.
        penalty: float = float(0.5*self.reg_strength * np.sum(weights * weights))
        #Combine classification and regularization terms
        loss: float = data_loss + penalty
        #Put +1 at violating rivals
        score_gradient: FloatArray = (margins > 0.0).astype(np.float64)
        #Count each sample's active rivals.
        violation_counts: FloatArray = score_gradient.sum(axis = 1)
        #Subtract all rivals at y.
        score_gradient[row_indices, encoded_labels] = -violation_counts
        #Chain rule to W
        weight_gradient: FloatArray = features.T @ score_gradient / batch_count
        #Add derivative of the L2 penalty.
        weight_gradient += self.reg_strength * weights
        #Chain rule to bias
        bias_gradient: FloatArray = score_gradient.sum(axis = 0) / batch_count
        #Supply objective and updates to the optimizer
        return loss, weight_gradient, bias_gradient

    def fit(self, features: ArrayLike, labels: ArrayLike) -> MulticlassLinearSVM:
        """Train from scratch, resetting any previously fiited state
        
        Parameters
        -----------
        features: ArrayLike, shape(N, D)
            Finite preprocessed feature rows: MNIST typically has D = 784
        labels: ArrayLike, shape(N, )
            Integer labels with at least two distinct classes

        Returns
        ---------
        MulticlassLinearSVM
            self, permitting model.fit(...).predict(.....).

        Raises
        --------
        ValueError
            Invalid features, labels, or fewer than two classes.
        FloatingPointError
            Numerical divergence; reduce the step size and check scaling.
        """
        #Validate the training matrix once.
        matrix: FloatArray = validate_features(features)
        #Validate row-aligned labels.
        targets: IntArray = validate_labels(labels, matrix.shape[0])
        #Annotate the two class-mapping arrays.
        classes: IntArray; encoded_labels: IntArray
        #Encode labels as 0....K-1
        classes, encoded_labels = np.unique(targets, return_inverse = True)
        if classes.size < 2: #A classifier requires competing for predictions
            raise ValueError("Training requires at least two distinct classes.")
        #Save the inverse mapping for predictions.
        self.classes_ = classes
        #Reset randomness.
        generator: np.random.Generator = np.random.default_rng(self.random_state)
        #Small initial W.
        self.weights_ = generator.normal(0.0, 0.001, (matrix.shape[1], classes.size))
        #Start all class intercepts at zero
        self.bias_ = np.zeros(classes.size, dtype = np.float64)
        #Discard history from a previous fit.
        self.loss_history_ = []
        #Type the epoch and batch-start loop variables
        epoch: int; start: int

        #Make complete passes over training samples
        for epoch in range(self.n_epochs):
            #Shuffle each epoch
            shuffled_indices: IndexArray = generator.permutation(matrix.shape[0])
            #Decay eta.
            step_size: float = self.learning_rate / (1.0 + self.learning_rate_decay * epoch)
            #Accumulate batch loss multiplied by batch size.
            weighted_loss: float = 0.0

            #Include the final short batch
            for start in range(0, matrix.shape[0], self.batch_size):
                #Select rows
                batch_indices: IndexArray = shuffled_indices[start: start + self.batch_size]
                #Gather features of shape (B, D)
                batch_features: FloatArray = matrix[batch_indices]
                #Gather encoded targets (B, )
                batch_labels: IntArray = encoded_labels[batch_indices]

                #Type outputs.
                batch_loss: float; weight_gradient: FloatArray; bias_gradient: FloatArray
                #Evaluate current batch and its subgradients.
                batch_loss, weight_gradient, bias_gradient = self._loss_and_gradients(
                    batch_features, batch_labels
                )
                
                #Detect numerical divergence instead of hiding hit.
                if not np.isfinite(batch_loss):
                    raise FloatingPointError("Nonfinite loss; check scaling and learning_rate.")
                #Move W opposite its subgradient.
                self.weights_ -= step_size * weight_gradient
                #Update unregularized intercepts
                self.bias_ -= step_size * bias_gradient
                #Weight short batches correctly
                weighted_loss += batch_loss * batch_indices.size

            #Check last update.
            if not np.isfinite(self.weights_).all() or not np.isfinite(self.bias_).all():
                raise FloatingPointError("Nonfinite parameters; lower learning_rate.")

            #Average the observed batch losses
            epoch_loss: float = weighted_loss / matrix.shape[0]
            #Retain progress for optional batch losses
            self.loss_history_.append(epoch_loss)

            #Print only when requested.
            if self.verbose:
                #Report progress
                print(f"Epoch {epoch + 1:03d} / {self.n_epochs}: batch loss = {epoch_loss:.6f}")

        #Allow estimator-style chaning after successful training
        return self
                
    def decision_funtion(self, features: ArrayLike) -> FloatArray:
        """
        Return raw scores (N, K) for feature row (N, D).

        Parameters: features is finite ArrayLike with the training feature count.
        Returns: FloatArray; column k corresponds to classes_[k]. Scores are NOT
        probabilities. Raise RuntimeError before fit of ValueError for bad input.
        """

        #Type the parameter references
        weights: FloatArray; bias: FloatArray; classes: IntArray
        #Ensure model state exists
        weights, bias, classes = self._require_fiited()
        #Validate inference features.
        matrix: FloatArray = validate_features(features)

        #Require the same feature definition as training.
        if matrix.shape[1] != weights.shape[0]:
            raise ValueError("Prediction feature count does not match training.")

        return matrix @ weights + bias #Score every sample against every class.

    def predict(self, features: ArrayLike) -> IntArray:
        """
        Map feature rows (N, D) to original integer labels (N,).

        Parameters: features is finite ArrayLike with D training features.
        Returns: IntArray of predicted class IDs. Exact ties select the first
        score column (the smallest original label). Propagates validation errors.
        """
        #Compute raw class evidence.
        scores: FloatArray = self.decision_funtion(features) 
        #Choose each row's maximum score.
        winning_indices: IndexArray = np.argmax(scores, axis = 1)
        #Type initializd model arrays.
        weights: FloatArray; bias: FloatArray; classes: IntArray
        #Retrieve the class-column mapping
        weights, bias, classes = self._require_fiited()
        #Restore original labels rather than returning encoded IDs.
        return classes[winning_indices]

    def score(self, features: ArrayLike, labels: ArrayLike) -> float:
        """
        Return classification accuracy in [0, 1]

        Parameters: features is ArrayLike (N, D); labels is integer ArrayLike(N, ).
        Returns: float, fraction of correctly predicted sample. Unseen target
        labels count as incorrect. Propagates feature/label validation errrors.
        """
        #Classify the evaluation samples
        predictions: IntArray = self.predict(features)
        #Check aligned ground truth
        targets: IntArray = validate_labels(labels, predictions.size)
        #Compute the correct-prediction fraction
        return float(np.mean(predictions == targets))
        
        

        
