from distutils.core import setup

setup(
    name='space_bandits',
    version='0.0.1',
    description='Contextual Bandits Library',
    author='Michael Klear',
    author_email='michael@launchpad.ai',
    install_requires=[
      'tensorflow>=1.5.0'
      'numpy>=1.14.3',
      'scipy>=0.19.1'
    ],
    packages=['space_bandits']
)