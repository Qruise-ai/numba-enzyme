"""
Module for primitive data types.
"""

from abc import abstractmethod

import numpy as np
from numba import types

Array = np.ndarray


class AbstractType:
    """
    Abstract Data types.
    """

    @abstractmethod
    def __call__(self):
        pass


class Number(AbstractType):
    """
    Abstract number type.

    Incorporates integeres and floats.
    """


class AbstractInt(Number):
    """
    Abstract integer data type.
    """


class Int32(AbstractInt):
    """
    Int32 data type.
    """

    def __new__(cls):
        """
        Create a new Numba int32 data type.

        Returns
        -------
        int32
            Numba int32 data type.
        """
        return types.int32


class Int64(AbstractInt):
    """
    Int64 data type.
    """

    def __new__(cls):
        """
        Create a new Numba int64 data type.

        Returns
        -------
        int64
            Numba int64 data type.
        """
        return types.int64


class AbstractFloat(Number):
    """
    Abstract float data type.
    """


class Float32(AbstractFloat):
    """
    Float32 data type.
    """

    def __new__(cls):
        """
        Create a new Numba float32 data type.

        Returns
        -------
        float32
            Numba float32 data type.
        """
        return types.float32


class Float64(AbstractFloat):
    """
    Float64 data type.
    """

    def __new__(cls):
        """
        Create a new Numba float64 data type.

        Returns
        -------
        float64
            Numba float64 data type.
        """
        return types.float64


class _ArrayFactory(type):
    """
    Factory class to create Numba data types.
    """

    def __call__(cls) -> types.abstract.ArrayCompatible:
        """
        Create associated Numba data type.

        Returns
        -------
        ArrayCompatible
            Numba data type of `ArrayCompatible` form.
        """
        eltype = cls.eltype()
        ndim = cls.ndim
        return types.Array(eltype, ndim, "C", False, aligned=True)


class AbstractArray(AbstractType):
    """
    Abstract array type.

    Attributes
    ----------
    eltype : Number
        Type of array's elements.
    ndim : int
        The rank of the array.
    n : int
        Total number of arguments in
        the LLVM representation.
    """

    eltype: Number = Int32
    ndim: int = 0
    n: int = 0


def ArrayND(T: Number, ndim: int):
    """
    Create a Metaclass for creating array data type.

    Parameters
    ----------
    T : AbstractType
        Data type of elements of the array.
    ndim : int
        The dimension of the array.

    Returns
    -------
    type
        Dynamically created type for
        an n-dimensional array with
        a given element type.
    """
    n = 5 + 2 * ndim
    return _ArrayFactory(
        f"Array{ndim}D_{T.__name__}",
        (AbstractArray,),
        {"eltype": T, "ndim": ndim, "n": n},
    )


def Array1D(T: Number):
    """
    Create a Metaclass for creating one-dimensional array.

    Parameters
    ----------
    T : Number
        Data type of elements of the array.

    Returns
    -------
    type
        Dynamically created type for
        a one-dimensional array with
        a given element type.
    """
    return ArrayND(T, 1)


def Array2D(T: Number):
    """
    Create a Metaclass for creating one-dimensional array.

    Parameters
    ----------
    T : Number
        Data type of elements of the array.

    Returns
    -------
    type
        Dynamically created type for
        a two-dimensional array with
        a given element type.
    """
    return ArrayND(T, 2)
