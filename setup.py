from setuptools import setup

setup(

    name="stan_utility",
    packages=[
        'stan_utility',
    ],
    version='v1.0a',
    license='BSD',
    description='Helper routines for pystan based off @betanalphas stan_utility',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
    #   url = 'https://github.com/grburgess/pychangcooper',
    #   download_url='https://github.com/grburgess/pychangcooper/archive/1.1.2.tar.gz',

    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'pystan',
        'pandas'
    ],
)
