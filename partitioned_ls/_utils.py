#!/usr/bin/env python
"""
Useful functions

Developer:
Omar Billotti

Description:
useful functions for the partitioned least squares algorithm
"""

from numpy import zeros, sum


def checkalpha(a, p):
    """
        checkalpha funtion

        Parameters
        ----------
        alpha : Array
        p : Matrix

        Returns
        -------
        out : Array
    """
    suma = sum(p * a.reshape(-1, 1), 0)
    sump = sum(p, 0)
    for k in range(0, p.shape[1]):
        if suma[k] == 0.0:
            for m in range(0, p.shape[0]):
                if p[m][k] == 1:
                    a[m] = 1.0 / sump[k]
    return a


def vec1(n):
    """
        Create an array with all elements a zero except the last element which
        will be the equal a 1.

        Parameters
        ----------
        n : integer

        Returns
        -------
        out : Array
               with all elements a zero except the last element which will be
               the equal a 1
    """
    result = zeros((1, n))
    result[0][n - 1] = 1
    return result


def indextobeta(b, k):
    """
       indextobeta funtion

        Parameters
        ----------
        b : integer

        k : integer

        Returns
        -------
        out : Array
              return 2 * bin(b,K) - 1 where bin(b,K) is a vector of K elements
              containing the binary representation of b.
    """
    result = []
    for i in range(0, k):
        result.append(2 * (b % 2) - 1)
        b = b >> 1
    return result


def bmatrix(x, p, beta):
    """
        bmatrix funtion

        Parameters
        ----------
        x : Matrix
        p : Matrix
        beta : Array

        Returns
        -------
        out : Matrix
              obtained multiplying each element in X to the associated
              weight in beta.
    """
    pbeta = p * beta
    featuremul = sum(pbeta, 1)
    return x * featuremul
