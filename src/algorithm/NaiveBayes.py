import numpy as np
# Define NaiveBayes
class NaiveBayes:
  def __init__(self, log_pdf):
    self.log_pdf = log_pdf #density function
  def fit(self,X,y):
    n_samples, n_features = X.shape
    self.classes = np.unique(y)
    n_classes = len(self.classes)
    self.mean = np.zeros((n_classes,n_features))
    self.var = np.zeros((n_classes,n_features))
    self.prior = np.zeros(n_classes)
    eps = 1e-9 * X.var(axis = 0).max() #avoid var = 0

    for idx, c in enumerate(self.classes):
      X_c = X[y == c]
      self.mean[idx] = X_c.mean(axis = 0)
      self.var[idx] = X_c.var(axis = 0) + eps
      self.prior[idx] = X_c.shape[0] /n_samples
    return self
  def predict(self,X):
    scores = []
    for idx in range(len(self.classes)):
      log_likelihood = np.sum(self.log_pdf(X,self.mean[idx],self.var[idx]),axis = 1)
      scores.append(log_likelihood + np.log(self.prior[idx]))
    return self.classes[np.argmax(scores,axis = 0)]
# dêfinition of log density function
def gaussian_log_pdf(X,mean,var):
  """logN(x_i|mean_i,var_i), shape(n_samples, n_features)"""
# X: (n_samples,n_features),mean and var: (n_features)
  return -0.5*np.log(2*np.pi*var) - (X-mean)**2/(2*var)

def multinomial_log_pdf(X, mean, var, alpha = 0.01):
  theta = (mean+alpha)/(np.sum(mean+alpha))
  return X*np.log(theta)

def bernouli_log_pdf(X,mean,var,alpha = 0.01):
  p = (mean + alpha) / (1 + 2*alpha)
  return X*np.log(p) + (1-X)*np.log(1-p)