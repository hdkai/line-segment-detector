# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cpython cimport array
from cython import boundscheck, wraparound, cdivision
from numpy cimport float64_t, int_t, ndarray
from numpy import arccos, arctan2, array, ceil, expand_dims
from numpy.linalg import eig, norm
from libc.math cimport atan2, cos, fmax, fmin, pow, sin, sqrt
from libcpp.vector cimport vector

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef array.array compute_rect (
    vector[coordinate] region,
    ndarray[float64_t, ndim=2, mode="c"] grad_norm
):
    """
    Compute a rectangle from a given pixel region.

    Parameters:
        region (ndarray): Region indices array with shape (N,2).
        grad_norm (ndarray): Gradient norm field with shape (H,W).

    Returns:
        ndarray: Line vector with shape (10,) containing [ x1, y1, x2, y2, xc, yc, l, w, a, θ ] 
    """
    # Compute centroid
    cdef int region_x = 0, region_y = 0
    cdef double center_x = 0, center_y = 0
    cdef int region_size = region.size()
    cdef double cell_grad = 0
    cdef double grad_sum = 0
    for i in range(region_size):
        region_y = region[i][0]
        region_x = region[i][1]
        cell_grad = grad_norm[region_y, region_x]
        center_x += region_x * cell_grad
        center_y += region_y * cell_grad
        grad_sum += cell_grad
    center_x /= grad_sum
    center_y /= grad_sum
    # Compute major and minor axes
    cdef double delta_x = 0, delta_y = 0
    cdef double m_xx = 0, m_yy = 0, m_xy = 0
    for i in range(region_size):
        region_y = region[i][0]
        region_x = region[i][1]
        delta_x = region_x - center_x
        delta_y = region_y - center_y
        cell_grad = grad_norm[region_y, region_x]
        m_xx += pow(delta_x, 2.) * cell_grad
        m_yy += pow(delta_y, 2.) * cell_grad
        m_xy += delta_x * delta_y * cell_grad
    m_xx /= grad_sum
    m_yy /= grad_sum
    m_xy /= grad_sum
    cdef double exp_term = sqrt(pow(m_xx - m_yy, 2.) + 4 * m_xy * m_xy)
    cdef double eig_1 = 0.5 * (m_xx + m_yy + exp_term)
    cdef double eig_2 = 0.5 * (m_xx + m_yy - exp_term)
    cdef double theta = atan2(eig_1 - m_xx, m_xy)
    cdef double dx = cos(theta)
    cdef double dy = sin(theta)
    # Compute length and width
    cdef double min_l = 0, min_w = 0, max_l = 0, max_w = 0
    for i in range(region_size):
        region_y = region[i][0]
        region_x = region[i][1]
        delta_x = region_x - center_x
        delta_y = region_y - center_y
        l = delta_x * dx + delta_y * dy
        w = delta_x * -dy + delta_y * dx
        min_l = fmin(min_l, l)
        max_l = fmax(max_l, l)
        min_w = fmin(min_w, w)
        max_w = fmax(max_w, w)
    # Compute min and max points
    cdef double x1 = center_x + min_l * dx
    cdef double y1 = center_y + min_l * dy
    cdef double x2 = center_x + max_l * dx
    cdef double y2 = center_y + max_l * dy
    cdef double length = max_l - min_l
    cdef double width = max_w - min_w
    # Return
    return array.array("d", [ x1, y1, x2, y2, center_x, center_y, length, width, region_size, theta ])