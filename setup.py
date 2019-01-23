"""
A toolkit for various purposes.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    with open(os.path.join(here, path), encoding='utf-8') as f:
        version_file = f.read()
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = version('multitarget/__init__.py')

setup(
    name='multitarget',
    version=VERSION,

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
        'Programming Language :: Python :: 3.7',
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
        'scikit-learn',
        'pandas',
        'numpy',
    ],

)