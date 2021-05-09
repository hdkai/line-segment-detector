# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from numpy import arccos, arctan2, array, ceil, expand_dims
from numpy.linalg import eig, norm

def compute_rect (region, grad_norm):
    """
    Compute a rectangle from a given pixel region.

    Parameters:
        region (ndarray): Region indices array with shape (N,2).
        grad_norm (ndarray): Gradient norm field with shape (H,W).

    Returns:
        ndarray: Line vector with shape (10,) containing [ x1, y1, x2, y2, xc, yc, l, w, a, θ ] 
    """
    # Compute centroid
    grads = expand_dims(grad_norm[region[:,0], region[:,1]], axis=1)
    region = region[:,[1,0]] # (y,x) -> (x,y)
    grad_sum = grads.sum()
    centroid = (grads * region).sum(axis=0, keepdims=True) / grad_sum
    # Compute major and minor axes
    rel_region = region - centroid
    m_xx, m_yy = ((rel_region) ** 2 * grads).sum(axis=0) / grad_sum
    m_xy = (rel_region.prod(axis=1, keepdims=True) * grads).sum() / grad_sum
    M = array([
        [m_xx, m_xy],
        [m_xy, m_yy]
    ])
    w, V = eig(M)
    major_axis, minor_axis = V.T[w.argmax()], V.T[w.argmin()]
    # Compute size and angle
    theta = arctan2(minor_axis[1], minor_axis[0])
    lengths, widths = rel_region @ major_axis, rel_region @ minor_axis
    length, width = ceil(lengths.max() - lengths.min()), ceil(widths.max() - widths.min())
    # Compute min and max points
    min_point = (centroid + lengths.min() * major_axis).squeeze()
    max_point = (centroid + lengths.max() * major_axis).squeeze()
    # Return stack: [ x1, y1, x2, y2, cx, cy, l, w, a, θ ]
    centroid = centroid.squeeze()
    aligned = region.shape[0]
    return array([ min_point[0], min_point[1], max_point[0], max_point[1], centroid[0], centroid[1], length, width, aligned, theta ])