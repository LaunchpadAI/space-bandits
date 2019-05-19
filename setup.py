from distutils.core import setup

long_desc = """
A practical library for building contextual bandits models with deep Bayesian approximation.
Supports both online learning and offline training of models as well as novel methods for cross-validating CB models on historic data.
"""

setup(
    name='space-bandits',
    version='0.0.95',
    description='Deep Bayesian Contextual Bandits Library',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael@fellowship.ai',
    url='https://github.com/fellowship/space-bandits',
    download_url='https://github.com/fellowship/space-bandits/archive/v0.0.95.tar.gz',
    install_requires=[
      'numpy>=1.14.5',
      'scipy>=0.19.1',
      'pandas>=0.21.0',
      'cython',
      'scikit-learn',
      'torch'
    ],
    packages=['space_bandits']
)
