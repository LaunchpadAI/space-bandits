from distutils.core import setup
import os

long_desc = """
A practical library for building contextual bandits models with deep Bayesian approximation.
Supports both online learning and offline training of models as well as novel methods for cross-validating CB models on historic data.
"""

reqs= [
    'torch',
    'numpy',
    'scipy',
    'pandas',
    'cython',
    'scikit-learn'
]
version = '0.0.993'

setup(
    name='space-bandits',
    version=f'{version}',
    description='Deep Bayesian Contextual Bandits Library',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael@fellowship.ai',
    url='https://github.com/fellowship/space-bandits',
    download_url=f'https://github.com/fellowship/space-bandits/archive/v{version}.tar.gz',
    install_requires=reqs,
    packages=['space_bandits']
)
