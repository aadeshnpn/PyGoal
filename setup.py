#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

from setuptools import setup, find_packages
from codecs import open

requires = [
    'numpy'
]

version = ''
with open('__init__.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)


setup(
    name='PyGoal',
    version=version,
    description="LTL based Goal Framework in Python 3+",
    long_description="testing",
    author='Project PyGoal Team',
    author_email='aadeshnpn@byu.net',
    url='https://github.com/aadeshnpn/PyGoal',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    keywords='LTL',
    license='Apache 2.0',
    zip_safe=False,
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Goal Specification',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Alpha',
        'Natural Language :: English',
    ],
)
