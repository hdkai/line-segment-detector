# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cv2 import filter2D, BORDER_REFLECT
from numpy import arctan2, array, stack
from numpy.linalg import norm

def compute_gradient_field (input):
    """
    Compute gradient field.

    Parameters:
        input (ndarray): Greyscale image with shape (H,W).

    Returns:
        tuple: Gradient vector field with shape (H,W,2) and gradient norm field with shape (H,W).
    """
    # Compute image gradient
    kernel_x = array([
        [-0.5, 0.5],
        [-0.5, 0.5]
    ])
    kernel_y = array([
        [-0.5, -0.5],
        [0.5, 0.5]
    ])
    grad_field = stack([
        filter2D(input, -1, kernel_x, anchor=(0, 0), borderType=BORDER_REFLECT),
        filter2D(input, -1, kernel_y, anchor=(0, 0), borderType=BORDER_REFLECT)
    ], axis=-1)
    # Compute LLA and norm
    grad_norm = norm(grad_field, axis=2)
    return grad_field, grad_norm