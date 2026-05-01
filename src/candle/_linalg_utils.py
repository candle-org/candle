"""Legacy Tensor linalg method helpers."""

from . import linalg


def solve(input, other):
    return linalg.solve(other, input)


def lstsq(input, other):
    return linalg.lstsq(other, input)


def eig(input, eigenvectors=False):
    values, vectors = linalg.eig(input)
    if eigenvectors:
        return values, vectors
    return values, None


def _symeig(input, eigenvectors=False):
    values, vectors = linalg.eigh(input)
    if eigenvectors:
        return values, vectors
    return values, None
