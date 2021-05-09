# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cpython cimport array
from cython import boundscheck, wraparound, cdivision
from libcpp.vector cimport vector
from numpy cimport float64_t, int_t, ndarray
from numpy import array, zeros_like

from .common cimport coordinate
from .rect cimport compute_rect
from .region cimport grow_region

@boundscheck(False)
@wraparound(False)
@cdivision(True)
def scan_segments (
    ndarray[int_t, ndim=2, mode="c"] coordinates,
    ndarray[float64_t, ndim=3, mode="c"] grad_field,
    ndarray[float64_t, ndim=2, mode="c"] grad_norm,
    double angle_tolerance,
    double rho
):
    # Create exploration log
    cdef ndarray[int_t, ndim=2, mode="c"] explored = zeros_like(grad_norm, dtype=int)
    cdef int height = grad_field.shape[0]
    cdef int width = grad_field.shape[1]
    explored[grad_norm <= rho] = True
    explored[0,0] = True
    explored[0,width-1] = True
    explored[height-1,0] = True
    explored[height-1,width-1] = True
    # Loop
    cdef coordinate seed
    cdef vector[coordinate] region
    cdef array.array rect
    cdef array.array result = array.array("d")
    for i in range(coordinates.shape[0]):
        # Check
        seed = (coordinates[i, 0], coordinates[i, 1])
        if explored[seed[0], seed[1]]:
            continue
        # Compute rect
        region = grow_region(seed, grad_field, angle_tolerance, explored)
        rect = compute_rect(region, grad_norm)
        array.extend(result, rect)
    return array(result).reshape(-1, 10)