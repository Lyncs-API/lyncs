import codecs, os
from setuptools import setup, find_packages
from lyncs_setuptools import find_long_description, get_data_files, find_version, classifiers

setup(
    name='lyncs_setuptools',
    version=find_version(),
    author='Simone Bacchio',
    author_email='s.bacchio@gmail.com',
    description='CMake installation tools for Lyncs',
    long_description=find_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='BSD',
    classifiers = classifiers,
    data_files=get_data_files(),
    keywords = [
        "Lyncs",
        "setuptools",
        "cmake",
        ],
    )
