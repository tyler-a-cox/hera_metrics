#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from copy import copy

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "scipy",
    "tqdm",
    "jax",
    "hera_filters @ git+http://github.com/HERA-Team/hera_filters",
]

test_requirements = copy(requirements)

setup(
    author="Tyler Cox",
    author_email="tyler.a.cox@berkeley.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python package for calibrating radio arrays using frequency redundancy",
    entry_points={"console_scripts": ["hera_metrics=hera_metrics.cli:main",],},
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords="hera_metrics",
    name="hera_metrics",
    packages=find_packages(include=["hera_metrics", "hera_metrics.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/tyler-a-cox/hera_metrics",
    version="0.1.0",
    zip_safe=False,
    dependency_links=[
        "git+http://github.com/HERA-Team/hera_filters",
    ],
)
