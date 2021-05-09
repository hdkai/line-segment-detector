# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from numpy import array, zeros_like

from .region import grow_region
from .rect import compute_rect

def scan_segments (coordinates, grad_field, grad_norm, angle_tolerance, rho):
    explored = zeros_like(grad_norm).astype(bool)
    explored[grad_norm <= rho] = True
    height, width = grad_field.shape[:2]
    explored[[0, 0, height-1, height-1], [0, width-1, 0, width-1]] = True
    result = []
    for seed in coordinates:
        # Check
        if explored[tuple(seed)]:
            continue
        # Compute rect
        region = grow_region(seed, grad_field, angle_tolerance, explored)
        rect = compute_rect(region, grad_norm)
        result.append(rect)
    return array(result)