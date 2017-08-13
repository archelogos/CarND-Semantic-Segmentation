from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy==1.13.1',
                    'scipy==0.19.1',
                    'tqdm==4.15.0',
                    'Pillow==4.2.1']

setup(
    name='FCN TRAINER',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='FCN TRAINER'
)
