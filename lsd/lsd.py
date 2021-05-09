# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cv2 import cvtColor, filter2D, GaussianBlur, resize, BORDER_REFLECT, COLOR_RGB2GRAY, INTER_AREA
from numpy import deg2rad, int32, log, ndarray, pi, stack, uint8, unravel_index, zeros_like
from numpy.linalg import norm

from .py import compute_gradient_field
from .c import scan_segments

def line_segment_detector (image: ndarray, scale=0.8, angle_tolerance=22.5):
    """
    Compute line segments in an image using a modified implementation of the 
    Line Segment Detector paper: http://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf

    Parameters:
        image (ndarray): Input image, either RGB or greyscale.
        scale (float): Downscale factor for processing.
        angle_tolerance (float): Angle tolerance in degrees.

    Returns:
        ndarray: Lines array with shape (N,8), with each row being [ x1, y1, x2, y2, cx, cy, l, w ].
    """
    # Input checking
    assert isinstance(image, ndarray), "Image must be a numpy.ndarray"
    image = cvtColor(image, COLOR_RGB2GRAY) if image.ndim == 3 else image
    image = image / 1. if image.dtype == uint8 else image * 255.
    # Scale image
    if scale < 1.:
        sigma = 0.6 / scale
        input = GaussianBlur(image, (0, 0), sigma)
        input = resize(input, (0, 0), fx=scale, fy=scale, interpolation=INTER_AREA)
    else:
        input = image
    # Compute image gradient
    grad_field, grad_norm = compute_gradient_field(input)
    angle_tolerance = deg2rad(angle_tolerance)
    rho = 2. / angle_tolerance
    precision = angle_tolerance / pi
    # Get order
    indices = grad_norm.argsort(axis=None)[::-1]
    coordinates = stack(unravel_index(indices, grad_norm.shape), axis=1).astype(int) # (y, x)
    # Scan
    result = scan_segments(coordinates, grad_field, grad_norm, angle_tolerance, rho)
    # Filter count
    min_size = -2.5 * (log(input.shape[0]) + log(input.shape[1])) / log(precision)
    result = result[result[:,8] > min_size]
    # Filter area and density
    MIN_AREA = 0
    area_mask = result[:,6] * result[:,7] > MIN_AREA
    result = result[area_mask]
    # Filter density
    MIN_DENSITY = 0.8
    density_mask = result[:,8] / result[:,6] * result[:,7] > MIN_DENSITY
    result = result[density_mask]
    # Return
    result = result[:,:8] / scale
    return result