#!/usr/bin/env python
"""
Partitioned Least Square class

Developer:
Omar Billotti

Description:
Partitioned Least Square class
"""
from numpy import shape, zeros, hstack, ones, vstack, sum as sum_elements, array, inf, where
from numpy.random import rand
from numpy.linalg import lstsq
from scipy.optimize import nnls
from scipy.linalg import norm

from ._utils import vec1, indextobeta, checkalpha, bmatrix


class PartitionedLs(object):
    """
    Partitioned Least Square class
    """

    def __init__(self, algorithm="alt"):
        """
            Constructor of Partioned Least Square Class

            Parameters
            ----------
            algorithm : string
                        String used to set some algorithm to choose to create
                        the model. possible values alt and opt


            Returns
            -------
            None.
        """
        self.model = None
        self.algorithm = algorithm

    def fit(self, x, y, p):
        """
            Fits a PartialLS Regression model to the given data

            Parameters
            ----------
            x : Matrix
                describing the examples
            y : Array
                vector with the output values for each example
            p : Matrix
                specifying how to partition the M attributes into K subsets.
                P{m,k} should be 1 if attribute number m belongs to partition k

            Returns
            -------
            None.
        """
        if self.algorithm == "opt":
            self.__fit_opt_nnls(x, y, p)
        elif self.algorithm == "alt":
            self.__fit_alt_nnls(x, y, p)
        else:
            self.__fit_alt_nnls(x, y, p)

    def __fit_opt_nnls(self, x,
                       y, p):
        """
            Fits a PartialLS OPT Regression model to the given data

            Parameters
            ----------
            x : Matrix
                describing the examples
            y : Array
                vector with the output values for each example
            p : Matrix
                specifying how to partition the M attributes into K subsets.
                P{m,k} should be 1 if attribute number m belongs to partition k

            Returns
            -------
            None.
        """
        xo = hstack((x, ones((shape(x)[0], 1))))
        po = vstack(
            (hstack((p, zeros((shape(p)[0], 1)))), vec1(shape(p)[1] + 1)))

        k = shape(po)[1]

        b_start, results = (-1, [])

        for i in range(b_start + 1, 2 ** k):
            beta = array(indextobeta(i, k))
            xb = bmatrix(xo, po, beta)
            alpha = nnls(xb, y)[0]
            optval = norm(xo.dot(po * alpha.reshape(-1, 1)).dot(beta) - y)

            result = (optval, alpha[:-1], beta[:-1], alpha[-1] * beta[-1], p)
            results.append(result)

        optvals = [r[0] for r in results]
        optindex = optvals.index(min(optvals))

        (opt, a, b, t, p) = results[optindex]

        A = sum_elements(p * a.reshape(-1, 1), 0)
        b = b * A

        # substituting all 0.0 with 1.0
        for z in where(A == 0.0):
            A[z] = 1.0

        a = sum_elements((p * a.reshape(-1, 1)) / A, 1)

        self.model = (opt, a, b, t, p)

    def __fit_alt_nnls(self, x,
                       y, p,
                       n=20):
        """
            Fits a PartialLS Alt Regression model to the given data

            Parameters
            ----------
            x : Matrix N * M
                matrix describing the examples
            y : vector
                vector with the output values for each example
            p : Matrix M * K
                specifying how to partition the M attributes into K subsets.
                P{m,k} should be 1 if attribute number m belongs to partition k
            n : int
                number of alternating loops to be performed, defaults to 20.

            Returns
            -------
            None.
        """

        # Rewriting the problem in homogenous coordinates
        xo = hstack((x, ones((shape(x)[0], 1))))
        po = vstack((hstack((p, zeros((shape(p)[0], 1)))),
                     vec1(shape(p)[1] + 1)))

        m, k = shape(po)

        alpha = rand(m)
        beta = (rand(k) - 0.5) * 10
        t = rand()

        initvals = (0, alpha, beta, t, inf)

        i_start, alpha, beta, t, optval = initvals

        for i in range(i_start + 1, n):
            # nnls problem with fixed beta variables
            po_beta = sum_elements(po * beta, 1)
            xo_beta = xo * po_beta
            alpha = nnls(xo_beta, y)[0]
            alpha = checkalpha(alpha, po)

            sum_alpha = sum_elements(po * alpha.reshape(-1, 1), 0)
            po_alpha = sum_elements(po * sum_alpha, 1)
            alpha = alpha / po_alpha
            beta = beta * sum_alpha

            # ls problem with fixed alpha variables
            xo_alpha = xo.dot(po * alpha.reshape(-1, 1))
            beta = lstsq(xo_alpha, y, rcond=None)[0]
            optval = norm(xo.dot(po * alpha.reshape(-1, 1)).dot(beta) - y, 2)

        self.model = (optval, alpha[:-1], beta[:-1], alpha[-1] * beta[-1], p)

    def predict(self, x):
        """
            Description
            Predicts points using the formula: f(X) = X * (P .* a) * b + t.

            Parameters
            ----------
            x : Matrix N * M
                    matrix describing the examples

            Returns
            -------
            out : Array
             contains the predictions of the given model on examples in X
        """
        (_, alpha, beta, t, p) = self.model
        return array(x).dot(p * alpha.reshape(-1, 1)).dot(beta) + t
