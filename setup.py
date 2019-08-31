#!/usr/bin/env python
from setuptools import setup, find_packages

# Always prefer setuptools over distutils
setup(
    name='doc2vec_agg',
    version='0.0.1',
    description='Generates simple document vectors from word2vec embeddings',
    url='https://github.com/TMiguelT/doc2vec_agg',
    author='Michael Milton',
    author_email='michael.r.milton@gmail.com',
    license='GPLv3',
    test_suite='test',
    packages=find_packages(include=['word2vec_agg']),
    install_requires=[
        'numpy',
        'scipy',
        'gensim'
    ]
)
