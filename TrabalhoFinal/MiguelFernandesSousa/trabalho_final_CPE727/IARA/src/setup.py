from setuptools import find_packages, setup

setup(
    name='iara',
    version='0.0.2',
    packages=find_packages(),
    description='package to access and train against the IARA datasets',
    author='Fabio Oliveira, Natanael Junior',
    license='Apache-2.0 License',
    install_requires=[],
    package_data={'iara': ['dataset_info/*.csv']},
)
