# 
#   Line Segment Detector
#   Copyright (c) 2021 Yusuf Olokoba.
#

from Cython.Build import cythonize
from Cython.Compiler import Options
from numpy import get_include
from setuptools import find_packages, setup, Extension

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="lsd-py",
    version="0.0.1",
    author="Yusuf Olokoba",
    author_email="yusuf@hdk.ai",
    description="Implementation of Line Segment Detector by Grompone von Gioi et al.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "cython>=0.29.21",
        "numpy",
        "opencv-python" # CHECK
    ],
    url="https://github.com/hdkai/Line-Segment-Detector",
    packages=find_packages(exclude=["examples", "test"]),
    ext_modules=cythonize(
        [
            Extension(
                "lsd.c.*",
                ["lsd/c/*.pyx"],
                include_dirs=[get_include()],
                extra_compile_args=["-O3"],
                language="c++"
                #define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ],
        compiler_directives={ "language_level": 3 },
        annotate=True
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ]
)