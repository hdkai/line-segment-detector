# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from libcpp.vector cimport vector
from numpy cimport float64_t, int_t, ndarray

from .common cimport coordinate

cpdef vector[coordinate] grow_region (
    #ndarray[int_t, ndim=1, mode="c"] seed,
    coordinate seed,
    ndarray[float64_t, ndim=3, mode="c"] grad_field,
    double tolerance,
    ndarray[int_t, ndim=2, mode="c"] explored
)