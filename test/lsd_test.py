# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from cv2 import GaussianBlur, imread, imwrite, line, resize, INTER_AREA
from numpy import array, deg2rad, uint8
from pytest import fixture, mark

from lsd import line_segment_detector
from lsd.py import compute_gradient_field

IMAGE_PATHS = [
    "media/1.jpg",
    "media/14.jpg"
]

@mark.parametrize("image_path", IMAGE_PATHS)
def test_downsample (image_path):
    image = imread(image_path, 0) / 1.
    scale = 0.8
    sigma = 0.6 / scale
    input = GaussianBlur(image, (0, 0), sigma)
    input = resize(input, (0, 0), fx=scale, fy=scale, interpolation=INTER_AREA)
    imwrite("downsample.jpg", input)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_grad_field (image_path):
    image = imread(image_path, 0) / 1.
    rho = 2. / deg2rad(22.5)
    grad_field, grad_norm = compute_gradient_field(image)
    grad_norm[grad_norm < rho] = 0.
    grad_norm = grad_norm.astype(uint8)
    imwrite("gradient.jpg", grad_norm)

@mark.parametrize("image_path", IMAGE_PATHS)
def test_lsd (image_path):
    image = imread(image_path, 1)
    lines = line_segment_detector(image, scale=0.6)
    # Render
    lines = lines[lines[:,6] > 20]
    for x1, y1, x2, y2 in lines[:,:4].astype(int):
        line(image, (x1, y1), (x2, y2), (0, 225, 255), 16)
    imwrite("lines.jpg", image)