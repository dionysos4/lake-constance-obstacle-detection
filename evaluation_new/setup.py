
from setuptools import setup, find_packages

setup(
    name='map_3d',
    version='1.0',
    author = "Dennis Griesser",
    license = "Apache License 2.0",
    description = ("Mean Average Precision with 3d IoU from pytorch 3d. The package is a fork from torchmetrics map"),
    install_requires=[
        'numpy >1.20.0',
        'torch >=1.8.1, <=2.0.1',
        'typing-extensions',
        'python_version < 3.9',
        'lightning-utilities >=0.7.0, <0.10.0'
    ],
    py_modules=['map_3d'],
    packages=find_packages(where="."),
    package_dir={"": "."},
)
