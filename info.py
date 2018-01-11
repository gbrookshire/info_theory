#!/usr/bin/env python

"""
A module to compute a few measures from information theory.
"""

from __future__ import division
import numpy as np
from pandas import qcut

from math import log, e

__license__ = "MIT"


def _bins(x):
    """ Get bin edges for an already-categorical variable
    """
    return np.append(np.unique(x), np.max(x) + 1)


def discrete(x, n):
    """ Sort a numeric variable x into n equally-populated bins.
    """
    return np.array(qcut(x, n, labels=range(1, n+1)))


def marginal_dist(x):
    """ Marginal distribution of a single (categorical) variable.
    """
    return np.histogram(x, _bins(x))[0] / x.size


def joint_dist(x, y):
    """ Joint distribution of two (categorical) variables.
    """
    Pxy = np.histogram2d(x, y, bins=(_bins(x), _bins(y)))[0] # Counts
    Pxy /= Pxy.sum() # Convert to proportions
    return Pxy


def _entropy_calc(p):
    """ Calculate entropy given a vector of probabilities
    """
    return -np.dot(p, np.log2(p))

# def entropy(x):
#     """ Shannon entropy of a (categorical) variable.
#         This version works, but is several times slower than the pure python
#         version below.
#     """
#     assert len(x.shape) == 1
#     p = marginal_dist(x)
#     H = _entropy_calc(p)
#     return H

def entropy(x):
  """ Computes entropy of label distribution. """

  n_labels = len(x)
  if n_labels <= 1: # H=0 for variables with only one outcome
    return 0

  # Get marginal distribution
  value,counts = np.unique(x, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1: # H=0 for variables with only one outcome
    return 0

  # Compute entropy
  ent = 0.
  for i in probs:
    ent -= i * log(i, 2)
  return ent


def joint_entropy(x,y):
    """ Shannon joint entropy
    """
    assert x.shape == y.shape
    Pxy = joint_dist(x, y)
    Pxy = np.ravel(Pxy) # Make into a vector
    Pxy = Pxy[Pxy != 0] # Exclude 0s because they'll confuse the log function
    H = _entropy_calc(Pxy)
    return H


def mutual_info(x, y):
    """
    Compute mutual information on two variables.
    Translated by GB from a Matlab function by Mo Chen (sth4nth@gmail.com)
    Checked against results from the python package dit. For simple integer
    vectors, these functions are a few times faster.

    Arguments
    x, y: numpy integer vectors of the same length
    """
    Hx = entropy(x)
    Hy = entropy(y)
    Hxy = joint_entropy(x, y)
    z = Hx + Hy - Hxy # Mutual information
    z = max(0, z);
    return z


def cross_mi(x, y, max_lag):
    """ Cross-mutual information. Analogous to cross-correlations, but replaces
    Pearson's correlation with mutual information as a function of time-lag.
    """
    z = []
    for n,lag in enumerate(range(-max_lag, max_lag + 1)):

        def _trim(s):
            """ Subfunction to trim off the shifted part of the sequence
            """
            # Cut off edges of a signal
            if lag > 0: # At positive lags, cut off shifted end of the signal
                return s[:-n]
            elif lag < 0: # At neg lags, cut off shifted beginnin of the signal
                return s[n:]
            else: # At 0 lag, don't change the signals
                return s

        # Shift one signal
        x_trim = np.roll(x, lag)
        # Cut off edges of both signals
        x_trim = _trim(x_trim)
        y_trim = _trim(y)
        # Compute MI
        z.append(mutual_info(x_trim, y_trim))

    return z
