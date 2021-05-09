# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from numpy import arctan2
from numpy.linalg import norm

from .region import grow_region
from .rect import compute_rect

def reduce_angle_tolerance (region, rect, seed, grad_field, explored):
    """
    Reduce angle tolerance, as described in the paper.

    Parameters:
        region (ndarray): INCOMPLETE.
        rect (ndarray): INCOMPLETE.
        seed (ndarray): INCOMPLETE.
        grad_norm (ndarray): INCOMPLETE.
        lla (ndarray): INCOMPLETE.
        explored (ndarray): INCOMPLETE.

    Returns:
        ndarray: INCOMPLETE.
    """
    # Reset region to not explored
    explored[region[:,0], region[:,1]] = False
    # Get region that is less than `width` away from rect center
    center, width = rect[4:6][::-1], rect[7]
    inliers = region[norm(region - center, axis=1) <= width]
    # Get LLA of new region
    inlier_grads = grad_field[inliers[:,0], inliers[:,1]]
    inlier_thetas = arctan2(inlier_grads[:,0], -inlier_grads[:,1])
    new_tolerance = 2. * inlier_thetas.std()
    # Compute new region and rect
    region = grow_region(seed, grad_field, new_tolerance, explored)
    return region

def reduce_region_radius (region, rect, grad_norm, explored):
    """
    Reduce region radius, as described in the paper.

    Parameters:
        region (ndarray): INCOMPLETE.
        rect (ndarray): INCOMPLETE.
        grad_norm (ndarray): INCOMPLETE.
        explored (ndarray): INCOMPLETE.

    Returns:
        tuple: INCOMPLETE.
    """
    # Get radius
    min_point, max_point, center = rect[0:2], rect[2:4], rect[4:6]
    len_min, len_max = norm(min_point - center), norm(max_point - center)
    radius = max(len_min, len_max)
    # Contract
    radius *= 0.75
    mask = norm(region - center, axis=1) <= radius
    inlier_region, outlier_region = region[mask], region[~mask]
    # Reset region to not explored
    explored[outlier_region[:,0], outlier_region[:,1]] = False
    # Recompute rect
    rect = compute_rect(inlier_region, grad_norm)
    return inlier_region, rect