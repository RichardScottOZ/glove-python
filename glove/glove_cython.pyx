#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3

import numpy as np
import scipy.sparse as sp
import collections
from cython.parallel import parallel, prange


cdef inline double double_min(double a, double b) nogil: return a if a <= b else b
cdef inline double double_max(double a, double b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
cdef inline int int_max(int a, int b) nogil: return a if a > b else b


cdef extern from "math.h" nogil:
    double sqrt(double)
    double c_log "log"(double)
    double fabs(double)
    double isfinite(double)


def fit_vectors(double[:, ::1] wordvec,
                double[:, ::1] wordvec_sum_gradients,
                double[::1] wordbias,
                double[::1] wordbias_sum_gradients,
                int[::1] row,
                int[::1] col,
                double[::1] counts,
                int[::1] shuffle_indices,
                double initial_learning_rate,
                double max_count,
                double alpha,
                double max_loss,
                int no_threads):
    """
    Estimate GloVe word embeddings given the cooccurrence matrix.
    Modifies the word vector and word bias array in-place.

    Training is performed via asynchronous stochastic gradient descent,
    using the AdaGrad per-coordinate learning rate.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_a, word_b
    cdef double count, learning_rate, gradient, gradient_sq
    cdef double log_count, ratio

    # Loss and gradient variables.
    cdef double prediction, entry_weight, loss
    
    # Temporary variables for better optimization
    cdef double temp_a, temp_b, weighted_loss
    cdef double eps = 1e-8  # Small constant for numerical stability

    # Iteration variables
    cdef int i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    with nogil:
        for j in prange(no_cooccurrences, num_threads=no_threads,
                        schedule='dynamic'):
            shuffle_index = shuffle_indices[j]
            word_a = row[shuffle_index]
            word_b = col[shuffle_index]
            count = counts[shuffle_index]
            
            # Skip invalid entries for robustness
            if count <= 0.0:
                continue

            # Compute log once for better efficiency
            log_count = c_log(count)

            # Get prediction (dot product)
            prediction = wordbias[word_a] + wordbias[word_b]
            for i in range(dim):
                prediction = prediction + wordvec[word_a, i] * wordvec[word_b, i]

            # Compute loss and the example weight.
            # Use more stable computation of the weight
            ratio = count / max_count
            if ratio > 1.0:
                entry_weight = 1.0
            else:
                entry_weight = ratio ** alpha
            
            # Compute weighted loss
            loss = prediction - log_count
            
            # Clip the loss for numerical stability (before weighting).
            if loss < -max_loss:
                loss = -max_loss
            elif loss > max_loss:
                loss = max_loss
            
            weighted_loss = entry_weight * loss

            # Update step: apply gradients using AdaGrad
            # Process word vectors with improved numerical stability
            # Note: Both updates use the pre-update values for symmetry
            for i in range(dim):
                # Save both values before updating (for symmetric gradient computation)
                temp_a = wordvec[word_a, i]
                temp_b = wordvec[word_b, i]
                
                # Update word_a
                gradient = weighted_loss * temp_b
                gradient_sq = gradient * gradient
                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_a, i] + eps)
                wordvec[word_a, i] = temp_a - learning_rate * gradient
                wordvec_sum_gradients[word_a, i] += gradient_sq

                # Update word_b (using pre-update temp_a)
                gradient = weighted_loss * temp_a
                gradient_sq = gradient * gradient
                learning_rate = initial_learning_rate / sqrt(wordvec_sum_gradients[word_b, i] + eps)
                wordvec[word_b, i] = temp_b - learning_rate * gradient
                wordvec_sum_gradients[word_b, i] += gradient_sq

            # Update word biases with improved numerical stability
            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_a] + eps)
            wordbias[word_a] -= learning_rate * weighted_loss
            wordbias_sum_gradients[word_a] += weighted_loss * weighted_loss

            learning_rate = initial_learning_rate / sqrt(wordbias_sum_gradients[word_b] + eps)
            wordbias[word_b] -= learning_rate * weighted_loss
            wordbias_sum_gradients[word_b] += weighted_loss * weighted_loss


def transform_paragraph(double[:, ::1] wordvec,
                        double[::1] wordbias,
                        double[::1] paragraphvec,
                        double[::1] sum_gradients,
                        int[::1] row,
                        double[::1] counts,
                        int[::1] shuffle_indices,
                        double initial_learning_rate,
                        double max_count,
                        double alpha,
                        int epochs):
    """
    Compute a vector representation of a paragraph. This has
    the effect of making the paragraph vector close to words
    that occur in it. The representation should be more
    similar to words that occur in it multiple times, and
    less close to words that are common in the corpus (have
    large word bias values).

    This should be be similar to a tf-idf weighting.
    """

    # Get number of latent dimensions and
    # number of cooccurrences.
    cdef int dim = wordvec.shape[1]
    cdef int no_cooccurrences = row.shape[0]

    # Hold indices of current words and
    # the cooccurrence count.
    cdef int word_b
    cdef double count, log_count, ratio
    cdef double gradient_sq, gradient_val

    # Loss and gradient variables.
    cdef double prediction
    cdef double entry_weight
    cdef double loss, weighted_loss
    cdef double gradient
    cdef double learning_rate
    cdef double eps = 1e-8  # Small constant for numerical stability

    # Iteration variables
    cdef int epoch, i, j, shuffle_index

    # We iterate over random indices to simulate
    # shuffling the cooccurrence matrix.
    for epoch in range(epochs):
        for j in range(no_cooccurrences):
            shuffle_index = shuffle_indices[j]

            word_b = row[shuffle_index]
            count = counts[shuffle_index]
            
            # Skip invalid entries
            if count <= 0.0:
                continue
            
            log_count = c_log(count)

            # Get prediction
            prediction = wordbias[word_b]
            for i in range(dim):
                prediction = prediction + paragraphvec[i] * wordvec[word_b, i]

            # Compute loss and the example weight with improved stability.
            ratio = count / max_count
            if ratio > 1.0:
                entry_weight = 1.0
            else:
                entry_weight = ratio ** alpha
            
            loss = entry_weight * (prediction - log_count)

            # Update step: apply gradients with improved numerical stability.
            for i in range(dim):
                learning_rate = initial_learning_rate / sqrt(sum_gradients[i] + eps)
                gradient = loss * wordvec[word_b, i]
                paragraphvec[i] = paragraphvec[i] - learning_rate * gradient
                sum_gradients[i] += gradient * gradient
