# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cpython cimport array
from libcpp.vector cimport vector
from numpy cimport float64_t, int_t, ndarray

from .common cimport coordinate

cpdef array.array compute_rect (
    vector[coordinate] region,
    ndarray[float64_t, ndim=2, mode="c"] grad_norm
)