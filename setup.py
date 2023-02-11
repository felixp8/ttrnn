import re
from setuptools import setup, find_packages
import sys
import os

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(name='ttrnn',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'gym',
          'matplotlib',
          'torch',
      ],
      description='ttRNN: a PyTorch package for task-trained RNNs',
      author='Felix Pei',
      url='https://github.com/felixp8/ttrnn/',
      version='0.0.1')