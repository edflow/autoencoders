from setuptools import setup, find_packages

setup(
    name='autoencoders',
    version='0.0.1',
    url='https://github.com/edflow/autoencoders.git',
    description='A collection of autoencoders in PyTorch',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'requests',
        'tqdm',
        'academictorrents'
    ],
)
