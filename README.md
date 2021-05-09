# Line Segment Detector
This is an open-source implementation of the [Line Segment Detector](http://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf) paper by Grompone von Gioi et al. This is a complete reimplementation from scratch, as the original source code publsihed with the paper is under the Aferro GPL license, which is much too restrictive for common use cases. As a result, this project is published under a much more permissive MIT license.

## Installing the Library
Pending release on PyPi, the library can be installed directly from GitHub:
```sh
pip3 install git+https://github.com/hdkai/Line-Segment-Detector.git
```

## Detecting Lines in an Image
The library exposes a single method:
```py
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
```
The function accepts a single color or greyscale image, and returns an `Nx8` matrix, where each row contains:
1. The `x` coordinate of the first point on the line segment.
2. The `y` coordinate of the first point on the line segment.
3. The `x` coordinate of the second point on the line segment.
4. The `y` coordinate of the second point on the line segment.
5. The `x` coordinate of the center of the line segment.
6. The `y` coordinate of the center of the line segment.
7. The length of the line segment `l`.
8. The width of the line segment `w`. For every detected line, `w < l`.

## Requirements
- Python 3.6+
- Cython 0.29.21+.