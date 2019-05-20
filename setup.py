from distutils.core import setup
import os

long_desc = """
A practical library for building contextual bandits models with deep Bayesian approximation.
Supports both online learning and offline training of models as well as novel methods for cross-validating CB models on historic data.
"""

def parse_requirements():
    """reads dependencies in from requirements.txt"""
    setup_dir = os.path.dirname(os.path.realpath(__file__))
    requirements_path = os.path.join(setup_dir, 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = f.read()
    return requirements.split('\n')

def parse_version():
    """gets current version of library"""
    setup_dir = os.path.dirname(os.path.realpath(__file__))
    version_path = os.path.join(setup_dir, 'space_bandits', '__init__.py')
    with open(version_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for line in lines:
        if "__version__" in line:
            version = line.split('=')[1]
    return version

version = parse_version()

setup(
    name='space-bandits',
    version=f'{version}',
    description='Deep Bayesian Contextual Bandits Library',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael@fellowship.ai',
    url='https://github.com/fellowship/space-bandits',
    download_url=f'https://github.com/fellowship/space-bandits/archive/v{version}.tar.gz',
    install_requires=parse_requirements(),
    packages=['space_bandits']
)
