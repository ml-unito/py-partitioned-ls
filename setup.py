#!/usr/bin/env python
"""py-partitioned-ls: Python implementation of algorithms for solving the Partitioned Least Squares problem
"""
from setuptools import setup

setup(
    name='py-partitioned-ls',
    version='0.0.1',
    author='Omar Billotti',
    author_email='o.billo95@gmail.com',
    packages=['partitioned_ls'],
    url='',
    license='MIT',
    description='Python implementation of algorithms for solving the'
                'Partitioned Least Squares problem',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "scipy",
    ],
)
