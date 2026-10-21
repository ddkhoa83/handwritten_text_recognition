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

    def __inti__(
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
        if not np.isfinite(reg_state) or reg_strength < 0:
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
        scores: FloatArray = X @ self.weights_ + self.bias

        #Convert scores to stable log-probabilities
        log_probabilities: FloatArray = self._log_softmax(scores)
        
         
        
        

    
        
            
    