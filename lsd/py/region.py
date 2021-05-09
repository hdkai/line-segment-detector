# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from numpy import abs, arccos, arctan2, array, cos, expand_dims, pi, sin
from numpy.linalg import norm

def grow_region (seed, grad_field, tolerance, explored):
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
    # Iterate
    grad_sum = grad_field[tuple(seed)]
    height, width = explored.shape
    region = []
    pending = [seed]
    neighbor_offsets = array([ [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1] ])
    while len(pending):
        # Get coordinate
        coord = pending.pop()
        # Check angle
        coord_grad = grad_field[tuple(coord)]
        coord_grad = coord_grad / norm(coord_grad)
        mean_grad = grad_sum / norm(grad_sum)
        delta_theta = arccos(mean_grad @ coord_grad)
        if delta_theta >= tolerance:
            continue
        # Mark
        region.append(coord)
        explored[tuple(coord)] = True
        grad_sum += coord_grad
        # Add neighborhood
        neighborhood = expand_dims(coord, axis=0) + neighbor_offsets
        neighborhood = neighborhood[
            (0 <= neighborhood[:,0]) &
            (0 <= neighborhood[:,1]) &
            (neighborhood[:,0] < height) &
            (neighborhood[:,1] < width)
        ]
        neighborhood = neighborhood[~explored[neighborhood[:,0], neighborhood[:,1]]]
        for neighbor in neighborhood:
            pending.append(neighbor)
    # Return
    return array(region)