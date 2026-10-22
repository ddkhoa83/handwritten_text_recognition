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

class LogisticRegression:
    """
    Multinomial logistic regression trained with mini-batch gradient descent.

    This class implements softmax regression for labels encoded as consecutive integers from 0 to n_classes - 1.

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

    Notes
    -------
    Each row of X represents one sample.
    This educational implementation expects finite numerical features and labels encoded as 0, 1,..., n_classes - 1.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_epochs: int = 30,
        batch_size: int = 256,
        reg_strength: float = 0.0001,
        random_state: int = 42,
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
        reg_strength: float
            Nonnegative L2 regularization coeffeicient
        random_state: int
            Seed used for reproducible shuffling.
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

        #Require a positive integer number of epochs.
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        #Reject a negative or nonfinite regularization coefficient.
        if not np.isfinite(reg_strength) or reg_strength < 0:
            raise ValueError("reg_strength must be finite and nonnegative.")

        #Store the learning rate as a Python float.
        self.learning_rate: float = float(learning_rate)

        #Store number of passes through the training data.
        self.n_epochs: int = n_epochs

        #Store the maximum number of samples per update.
        self.batch_size: int = batch_size

        #Store the strength of the weight penalty
        self.reg_strength: float = float(reg_strength)

        #Store the seed used to reproduce the training sample order.
        self.random_state: int = random_state

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

    @staticmethod
    def _log_softmax(scores: FloatArray) -> FloatArray:
        """
        Calculate numerically stable log-probabilities for each sample.

        Parameters
        ----------
        scores : NDArray[np.float64]
            Finite class scores with shape (n_samples, n_classes)

        Returns
        ---------
        NDArray[np.float64]
            Logarithms of softmax probabilities, with the same shape as scores.
            Exponentiating these values produces probabilities whose rows sum to approximately one.
        """
        #Find the largest score in each row.
        #keepdims = True preserves shape (n_samples, 1) for broadcasting
        row_maximum: FloatArray = np.max(
            scores,
            axis = 1,
            keepdims = True,
        )

        #Subtract each row's maximum from all scores in that row.
        #This leaves softmax probabilities unchanged and prevents exp overflow.
        shifted_scores: FloatArray = scores - row_maximum

        #Exponentiate the shifted scores
        #All shifted scores are <= 0, so these exponentials are <= 1.
        exponential_scores: FloatArray = np.exp(shifted_scores)

        #Add the exponential values across classes for each sample
        #At least one exponential per row is 1, so the sum is positive.
        exponential_sums: FloatArray = np.sum(
            exponential_scores,
            axis = 1,
            keepdims = True,
        )

        #Calculate log(sum(exp(shifted_scores))) separately for every row.
        log_normalizer: FloatArray = np.log(exponential_sums)

        #Compute log-softmax directly
        #This avoids taking log of a probability that may underflow to zero.
        log_probabilities: FloatArray = shifted_scores - log_normalizer

        #Return one log-probability for every sample-class combination.
        return log_probabilities
    def _check_is_fitted(self) -> None:
        """
        Check whether the model parameters have been initialized

        Parameters
        -----------
        None

        Returns
        ----------
        None
            Returns normally when parameters exist.

        Raises
        ---------
        RuntimeError
            If fit has not initialized the model.
        """
        #All three attributes must be available before perdiction
        if(
            self.weights_ is None
            or self.bias_ is None
            or self.classes_ is None
        ):
            raise RuntimeError("Call fit before using this method.")

    def _loss_and_gradients(
        self,
        X: FloatArray,
        y: LabelArray,
    ) -> tuple[float, FloatArray, FloatArray]:
        """
        Compute the regularized loss and gradients for one mini-batch.

        Parameters
        ------------
        X: NDArray[np.float64]
            Feature matrix with shape (batch_size, n_features).
        y: NDArray[np.int64]
            Correct class indices with shape (batch_size,).
        Returns
        -----------
        loss: float
            Mean cross-entropy plus the L2 weight penalty.
        weight_gradient: NDArray[np.float64]
            Gradient with respect to weights, shape (n_features, n_classes).
        bias_gradient: NDArray[np.float64]
            Gradient with respect to biasses, shape (n_classes, ).
        """

        #Ensure model parameters are available
        self._check_is_fitted()

        #These assertions also help type checkers narrow optional attributes.
        assert self.weights_ is not None
        assert self.bias_ is not None

        #Read the actual batch size.
        #The last batch may be smaller than the configured batch_size
        n_samples: int = X.shape[0]

        #Calculate class scores using matrix multiplication and broadcasting
        scores: FloatArray = X @ self.weights_ + self.bias_

        #Convert scores to stable log-probabilities
        log_probabilities: FloatArray = self._log_softmax(scores)

        #Create the row indices 0, 1, ..., B - 1.
        row_indices: NDArray[np.intp] = np.arange(n_samples)

        #Select each sample's correct-class log-probability.
        #Take the negative mean to obtain cross-entropy
        data_loss: float = float(
            -np.mean(log_probabilities[row_indices, y])
        )
        
        #Calculate the L2 penalty on weights only
        regularization_loss: float = float(
            0.5 * self.reg_strength * np.sum(self.weights_ ** 2)
        )

        #Add th data loss and regularization penalty.
        loss: float = data_loss + regularization_loss

        #Convert log-probabilities into ordinary probabilities.
        probabilities: FloatArray = np.exp(log_probabilities)

        #Copy probabilities because the next line modifies this array.
        errors: FloatArray = probabilities.copy()

        #Subtract one from the correct-class entry of each sample
        #This calculates P - Y without explicitly building one-hot labels
        errors[row_indices, y] -= 1

        #Calculate the mean data gradient and add the weight penalty gradient
        #(d, B)@(B, C) produces a gradient of the shape (d, C)
        weight_gradient: FloatArray = (
            X.T @ errors / n_samples + self.reg_strength*self.weights_
        )

        #Average errors over samples to obtain one gradient per class bias.
        #No regularization term is added because biases are not penalized
        bias_gradient: FloatArray = np.mean(errors, axis = 0)

        #Return the objective value and both parameter gradients
        return loss, weight_gradient, bias_gradient

    def _training_loss(
        self,
        X: FloatArray,
        y: LabelArray,
    ) -> float:
        """
        Evaluate the full training objective using batches.

        Parameters
        -----------
        X: NDArray[np.float64]
            Training features with shape (n_samples, n_features)
        y: NDArray[np.int64]
            Training labels with shape (n_samples, ).

        Returns
        ----------
        float
            Mean cross-entropy across all samples plus one L2 weight penalty.

        Notes
        ---------
        Batches limit the size of temporary score arrays.
        Every batch is evaluated with the same current model parameters
        """
        #Ensure model parameters exist.
        self._check_is_fitted()

        #Narrow the optional parameter types
        assert self.weights_ is not None
        assert self.bias_ is not None

        #Count the total number of traning samples
        n_samples: int = X.shape[0]

        #Accumulate the sum, rather than mean, of sample losses
        total_data_loss: float = 0.0

        #Visit consecutive batches without changing the model
        for start in range(0, n_samples, self.batch_size):

            #Limit the end index so it never exceeds the dataset size.
            stop: int = min(start + self.batch_size, n_samples)

            #Select the current batch's feature rows
            X_batch: FloatArray = X[start:stop]

            #Select the matching class labels.
            y_batch: LabelArray = y[start:stop]

            #Compute class scores for the current batch.
            scores: FloatArray = X_batch @ self.weights_ + self.bias_

            #Calculate stable log_probabilities.
            log_probabilities: FloatArray = self._log_softmax(scores)

            #Build row indices for this particular batch.
            rows: NDArray[np.intp] = np.arange(stop - start)

            #Accumulate negative correct-class log-probabilities.
            total_data_loss -= float(
                np.sum(log_probabilities[rows, y_batch])
            )
                
        #Devide by the total number of samples to obtain the mean data loss.
        mean_data_loss: float = total_data_loss / n_samples

        #Calculate the weight penalty once for the entire objective.
        regularization_loss: float = float(
            0.5*self.reg_strength * np.sum(self.weights_ ** 2)
        )

        #Return the objective evaluated at the current parameters
        return mean_data_loss + regularization_loss

    def fit(
        self,
        X: FeatureArray | FloatArray,
        y: LabelArray,
    ) -> "LogisticRegression":
        """
        Train the model using shuffled mini-batch gradient descent.

        Parameters
        ----------
        X:  NDArray[np.float32] | NDArray[np.float64]
            Finite training features with shape (n_samples, n_features).
            For preporcessed MNIST, n_features is 784.
        y:  NDArray[np.int64]
            Integer labels with shape (n_samples,).
            All classes must be present and encoded as 0, ..., n_classes - 1.

        Returns
        --------
        LogisticRegresion
            The fitted object itself, allowing calls such as model.fit(x, y).

        Notes
        ------
        Calling fit again resets the weights, biases, and loss history
        """
        #Convert features to float64 for the numerical calculations.
        #This allocates a new array when the input uses float32
        X_array: FloatArray = np.asarray(X, dtype = np.float64)

        #Obtain a Numpy representation before validating the label dtypes
        y_array = np.asarray(y)

        #Require a nonempty two-dimension features matrix
        if(
            X_array.ndim != 2
            or X_array.shape[0] == 0
            or X_array.shape[1] == 0
        ):
            raise ValueError("X must be nonempty two-dimension array.")

        #Require one label per sample.
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must be one-dimensional with one label per row.")

        #Reject NaN and infinite feature values.
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values.")

        #Reject fractional labels instead of silently converting them.
        if not np.issubdtype(y_array.dtype, np.integer):
            raise ValueError("y must contain integer class labels.")

        #Convert valid integer labels into the dtype used by this class.
        y_array: LabelArray = y_array.astype(np.int64, copy = False)

        #Find the distinct class labels in sorted order
        classes: LabelArray = np.unique(y_array)

        #Determine the number of output classes.
        n_classes: int = len(classes)

        #Require at least two classes to train a classifier.
        if n_classes < 2:
            raise ValueError("Training data must contain at least two classes.")

        #Require consecutive labels because they directly index score columns.
        if not np.array_equal(classes, np.arange(n_classes)):
            raise ValueError("Labels must be encoded as 0, 1, ..., K- 1.")

        #Read the number of samples and input features
        n_samples, n_features = X_array.shape

        #Save th validated class labels
        self.classes_ = classes

        #Initialize one weight per features-class combination
        #Zero initialization is valid for this linear softmax model.
        self.weights_ = np.zeros(
            (n_features, n_classes),
            dtype = np.float64,
        )

        #Initialize one bias value per class.
        self.bias_ = np.zeros(n_classes, dtype = np.float64)

        #Clear any loss values left over from an earlier fit call.
        self.loss_history_ = []

        #Create an independent random generator for reproducible shuffling.
        random_generator: np.random.Generator = np.random.default_rng(
            self.random_state
        )

        #Perform n_epochs complete passes through the training datasets.
        for epoch in range(self.n_epochs):

            #Generate a shuffled ordering of all training sample indices.
            shuffled_indices: NDArray[np.intp] = (
                random_generator.permutation(n_samples)
            )

            #Move through the shuffled indices one mini-batch at a time.
            for start in range(0, n_samples, self.batch_size):

                #Select the sample indices belonging to this batch.
                batch_indices: NDArray[np.intp] = shuffled_indices[
                    start:start + self.batch_size
                ]

                #Select features rows in the shuffled batch order.
                X_batch: FloatArray = X_array[batch_indices]

                #Select the corresponding labels in the same order
                y_batch: LabelArray = y_array[batch_indices]

                #Calculate the gradients at the current parameter values.
                _, weight_gradient, bias_gradient = (
                    self._loss_and_gradients(X_batch, y_batch)
                )

                #Move weights in the direction opposite to their gradient.
                self.weights_ -= self.learning_rate*weight_gradient

                #Update biases using their own gradient.
                self.bias_ -= self.learning_rate*bias_gradient

            #Measure the full training objectie after the completed epoch.
            epoch_loss: float = self._training_loss(X_array, y_array)

            #Stop with an explicit error if numerical divergence occurs.
            if not np.isfinite(epoch_loss):
                raise FloatingPointError(
                    "Training loss is not finite. Try a smaller learning rate."
                )

            #Save this epoch's objective value for plotting and inspection.
            self.loss_history_.append(epoch_loss)

            #Print progress only when requested.
            if self.verbose:

                #Display an epoch number starting at one and six decimal places.
                print(
                    f"Epoch {epoch + 1:03d}/{self.n_epochs:03d} "
                    f"| Training loss: {epoch_loss: .6f}"
                )
        #Return the trained instance
        return self

    def predict_proba(
        self,
        X: FeatureArray | FloatArray,
    ) -> FloatArray:
        """
        Predict class probabilities for input samples

        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Finite features of shape (n_samples, n_features).
            Features must use the same preprocessing as the training data.

        Returns
        ---------
        NDArray[np.float64]
            Probabilities of shape (n_samples, n_classes).
            Column j corresponds to classes_[j]
            Each row sums to approximately one
        """

        #Reject prediction attemps made before fitting
        self._check_is_fitted()

        #Narrow the optional parameter types.
        assert self.weights_ is not None
        assert self.bias_ is not None

        #Convert input features to the numerical type used by the model.
        X_array: FloatArray = np.asarray(X, dtype = np.float64)

        #Require a two-dimensional matrix with the training feature count.
        if X_array.ndim != 2 or X_array.shape[1] != self.weights_.shape[0]:
            raise ValueError("X must have the same feature count as training.")

        #Reject invalid numerical inputs.
        if not np.isfinite(X_array).all():
            raise ValueError("X must contain only finite values.")

        #Calculate one score per sample and class
        scores: FloatArray = X_array @ self.weights_ + self.bias_

        #Obtain stable log-probabilities.
        log_probabilities: FloatArray = self._log_softmax(scores)

        #Exponentiate to obtain ordinary clas probabilities.
        probabilities: FloatArray = np.exp(log_probabilities)

        #Return the complete probability matrix.
        return probabilities

    def predict(
        self,
        X: FeatureArray | FloatArray,
    ) -> LabelArray:
        """Predict one class label for each input sample.
        
        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Feature matrix of shape (n_samples, n_features).

        Returns
        ---------
        NDArray[np.int64]
            Predicted class labels with shape (n_samples,).
            If probabilities tie, the first maximum is selected.
        """
        #Obtain the probability of every class for every sample.
        probabilities: FloatArray = self.predict_proba(X)

        #Find the highest - probability column in each row
        class_indices: NDArray[np.intp] = np.argmax(probabilities, axis = 1)

        #Convert probability-column indices into their corresponding labels.
        predicted_label: LabelArray = self.classes_[class_indices]

        #Return one predicted label per sample
        return predicted_label
        
    def score(
        self,
        X: FeatureArray | FloatArray,
        y: LabelArray,
    ) -> float:
        """
        Calculate classification accuracy.

        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Feature matrix of shape (n_samples, n_features).
        y : NDArray[np.int64]
            True class labels with shape (n_sample,).

        Returns
        --------
        float
            Fraction of correctly classified sample, between 0.0 and 1.0.
        """

        #Predict labels for the supplied features.
        predict_labels: LabelArray = self.predict(X)

        #Obtain an array representation of the true labels
        true_labels: LabelArray = np.asarray(y, dtype = np.int64)

        #Require matching, nonempty prediction and label arrays.
        if (
            true_labels.shape != predict_labels.shape
            or true_labels.size == 0
        ):
            raise ValueError("y must contain one label per sample and be nonempty.")

        #Compare predictions with labels element by element.
        #np.mean treats True as 1 and False as 0.
        accuracy: float = float(np.mean(predict_labels == true_labels))
        #Return the fraction of correct predictions
        return accuracy
            
        
        

           

        

    
        
        
        

    
        
            
    