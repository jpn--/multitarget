"""
A toolkit for various purposes.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='multitarget',
    version='1.0',

    description='Add-on tools for multi-target Gaussian process regression with scikit-learn',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jpn--/multitarget',

    # Author details
    author='Jeffrey Newman',
    author_email='jnewman@camsys.com',

    # Choose your license
    license='GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='multi-target gaussian process regression',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html

    install_requires=[
        'scikit-learn>=0.19.1',
        'pandas>=0.22.0',
        'numpy>=1.14',
    ],

)