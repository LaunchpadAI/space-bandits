from distutils.core import setup

setup(
    name='space_bandits',
    version='0.0.8',
    description='Deep Bayesian Contextual Bandits Library',
    author='Michael Klear',
    author_email='michael@launchpad.ai',
    url='https://github.com/AlliedToasters/space_bandits/archive/v0.0.8.tar.gz',
    install_requires=[
      'tensorflow>=1.5.0',
      'numpy>=1.14.3',
      'scipy>=0.19.1',
      'pandas>=0.21.0'
    ],
    packages=['space_bandits']
)