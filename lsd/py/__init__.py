# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from .density import reduce_angle_tolerance, reduce_region_radius
from .gradient import compute_gradient_field
from .rect import compute_rect
from .region import grow_region
from .scan import scan_segments