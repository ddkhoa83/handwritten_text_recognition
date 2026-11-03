
import numpy as np
class NaiveBayes:
    """
    A generic Naive Bayes classifier supporting different feature distributions.

    The classifier estimates class priors and class-specific feature statistics
    from the training data. A user-defined log-density function computes the
    log-likelihood of each feature given a class.

    Parameters
    ----------
    log_pdf : callable
        A function that computes the log-probability or log-density of each
        feature. It must accept X, mean, and var as arguments and return an
        array of shape (n_samples, n_features).

    Attributes
    ----------
    classes : NDArray
        Unique class labels observed in the training data.
    mean : NDArray[np.float64]
        Class-specific feature means of shape (n_classes, n_features).
    var : NDArray[np.float64]
        Class-specific feature variances of shape (n_classes, n_features).
    prior : NDArray[np.float64]
        Prior probabilities of the classes, with shape (n_classes).
    """

    def __init__(self, log_pdf):
        """
        Initialize the Naive Bayes classifier.

        Parameters
        ----------
        log_pdf : callable
            The log-density function used to calculate feature log-likelihoods.
        """
        self.log_pdf = log_pdf

    def fit(self, X, y):
        """
        Fit the classifier using the training data.

        Estimate the feature means, variances, and prior probabilities
        for each class.

        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Training features of shape (n_samples, n_features).
        y : NDArray
            Class labels of shape (n_samples,).

        Returns
        -------
        NaiveBayes
            The fitted classifier.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)

        # Add a small constant to prevent zero variances.
        eps = 1e-9 * max(float(X.var(axis=0).max()), 1.0)

        for idx, c in enumerate(self.classes):
            # Select the training samples belonging to the current class.
            X_c = X[y == c]

            # Estimate the feature statistics for the current class.
            self.mean[idx] = X_c.mean(axis=0)
            self.var[idx] = X_c.var(axis=0) + eps

            # Estimate the prior probability of the current class.
            self.prior[idx] = X_c.shape[0] / n_samples

        return self

    def predict(self, X):
        """
        Predict the class label for each input sample.

        The predicted class is the one with the highest log-posterior score.

        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Input features of shape (n_samples, n_features).
            The features must use the same preprocessing as the training data.

        Returns
        -------
        NDArray
            Predicted class labels of shape (n_samples,).
        """
        scores = []

        for idx in range(len(self.classes)):
            # Sum the feature log-likelihoods under the current class.
            log_likelihood = np.sum(
                self.log_pdf(X, self.mean[idx], self.var[idx]),
                axis=1
            )

            # Apply Bayes' rule in log space.
            scores.append(log_likelihood + np.log(self.prior[idx]))

        scores = np.array(scores).T

        # Return the class with the highest score for each sample.
        return self.classes[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        The probabilities are computed by applying the softmax function
        to the log-posterior scores.

        Parameters
        ----------
        X : NDArray[np.float32] | NDArray[np.float64]
            Input features of shape (n_samples, n_features).
            The features must use the same preprocessing as the training data.

        Returns
        -------
        NDArray[np.float64]
            Probabilities of shape (n_samples, n_classes).
            Column j corresponds to classes[j].
            Each row sums to approximately one.
        """
        scores = []

        for idx in range(len(self.classes)):
            # Calculate the log-likelihood of each sample for this class.
            log_likelihood = np.sum(
                self.log_pdf(X, self.mean[idx], self.var[idx]),
                axis=1
            )

            # Add the log prior to obtain the log-posterior score.
            scores.append(log_likelihood + np.log(self.prior[idx]))

        # Convert the scores to shape (n_samples, n_classes).
        scores = np.array(scores).T

        # Apply numerically stable softmax normalization.
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)

        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def gaussian_log_pdf(X, mean, var):
    """
    Compute the Gaussian log-density for each feature.

    Assumes that features are conditionally independent given the class
    and that each feature follows a Gaussian distribution.

    Parameters
    ----------
    X : NDArray[np.float32] | NDArray[np.float64]
        Input features of shape (n_samples, n_features).
    mean : NDArray[np.float64]
        Feature means of shape (n_features,).
    var : NDArray[np.float64]
        Feature variances of shape (n_features,). All values must be positive.

    Returns
    -------
    NDArray[np.float64]
        Feature-wise log-density values of shape (n_samples, n_features).
    """
    return (
        -0.5 * np.log(2 * np.pi * var)
        - (X - mean) ** 2 / (2 * var)
    )


def multinomial_log_pdf(X, mean, var, alpha=0.01):
    """
    Compute the Multinomial Naive Bayes feature log-likelihoods.

    The feature probabilities are estimated from the class-specific means
    using additive (Laplace-style) smoothing.

    Parameters
    ----------
    X : NDArray[np.float32] | NDArray[np.float64]
        Non-negative feature counts of shape (n_samples, n_features).
    mean : NDArray[np.float64]
        Class-specific feature means of shape (n_features,).
    var : NDArray[np.float64]
        Class-specific feature variances. This parameter is unused but is
        retained for compatibility with the classifier interface.
    alpha : float, default=0.01
        Positive smoothing parameter used to avoid zero probabilities.

    Returns
    -------
    NDArray[np.float64]
        Feature-wise log-likelihood contributions of shape
        (n_samples, n_features).
    """
    # Estimate smoothed feature probabilities.
    smoothed_mean = mean + alpha
    theta = smoothed_mean / np.sum(smoothed_mean)

    # Compute the feature-wise log-likelihood contributions.
    return X * np.log(theta)


def bernoulli_log_pdf(X, mean, var, alpha=0.01):
    """
    Compute the Bernoulli Naive Bayes feature log-likelihoods.

    Each feature is assumed to be binary, taking values of either zero
    or one. Additive smoothing is applied to estimate the probability
    that each feature is present.

    Parameters
    ----------
    X : NDArray[np.float32] | NDArray[np.float64]
        Binary input features of shape (n_samples, n_features).
    mean : NDArray[np.float64]
        Class-specific feature means of shape (n_features,).
        Each mean represents the estimated proportion of ones.
    var : NDArray[np.float64]
        Class-specific feature variances. This parameter is unused but is
        retained for compatibility with the classifier interface.
    alpha : float, default=0.01
        Positive smoothing parameter used to avoid probabilities of zero
        or one.

    Returns
    -------
    NDArray[np.float64]
        Feature-wise log-likelihood contributions of shape
        (n_samples, n_features).
    """
    # Estimate smoothed probabilities of feature presence.
    p = (mean + alpha) / (1 + 2 * alpha)

    # Compute the log-likelihood for both present and absent features.
    return X * np.log(p) + (1 - X) * np.log(1 - p)
