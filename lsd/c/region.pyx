# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cython import boundscheck, wraparound, cdivision
from numpy cimport float64_t, import_array, int_t, ndarray
from numpy import arccos, array, expand_dims, int
from numpy.linalg import norm
from libc.math cimport acos, sqrt
from libcpp.vector cimport vector

import_array()

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef vector[coordinate] grow_region (
    coordinate seed,
    ndarray[float64_t, ndim=3, mode="c"] grad_field,
    double tolerance,
    ndarray[int_t, ndim=2, mode="c"] explored
):
    """
    Grow a region of pixels from a given seed coordinate.

    Parameters:
        seed (ndarray): Seed index with shape (2,).
        grad_field (ndarray): Gradient field with shape (H,W).
        tolerance (float): Angle tolerance in radians.
        explored (ndarray): Exploration log with shape (H,W).

    Returns:
        ndarray: Region indices array with shape (N,2).
    """
    # Definitions
    cdef coordinate coord = seed, ncoord
    cdef (double, double) region_grad = (0, 0), region_dir
    cdef double region_norm
    cdef (double, double) grad, grad_dir
    cdef double grad_norm
    cdef double delta_theta
    cdef int height = explored.shape[0]
    cdef int width = explored.shape[1]
    cdef int[8] x_offsets = [-1, 0, 1, -1, 1, -1, 0, 1]
    cdef int[8] y_offsets = [-1, -1, -1, 0, 0, 1, 1, 1]
    # Iterate
    cdef vector[coordinate] region, pending
    pending.push_back(coord)
    while not pending.empty():
        # Get coordinate
        coord = pending.back()
        pending.pop_back()
        if explored[coord[0], coord[1]]:
            continue
        # Get grad direction
        grad = (grad_field[coord[0], coord[1], 1], grad_field[coord[0], coord[1], 0])
        grad_norm = sqrt(grad[0] * grad[0] + grad[1] * grad[1])
        grad_dir = (grad[0] / grad_norm, grad[1] / grad_norm)
        # Check angle
        region_norm = sqrt(region_grad[0] * region_grad[0] + region_grad[1] * region_grad[1])
        region_dir = (region_grad[0] / region_norm, region_grad[1] / region_norm)
        delta_theta = acos(region_dir[0] * grad_dir[0] + region_dir[1] * grad_dir[1])
        if delta_theta >= tolerance:
            continue
        # Mark
        region.push_back(coord)
        explored[coord[0], coord[1]] = True
        region_grad = (region_grad[0] + grad[0], region_grad[1] + grad[1])
        # Add neighborhood
        for i in range(8):
            ncoord = (coord[0] + y_offsets[i], coord[1] + x_offsets[i])
            if 0 <= ncoord[0] < height and 0 <= ncoord[1] < width and explored[ncoord[0], ncoord[1]] == 0:
                pending.push_back(ncoord)
    # Return
    return region