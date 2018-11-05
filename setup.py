from setuptools import setup
import os



stan_cache = os.path.expanduser('~/.stan_cache')

def ensure_dir(file_path):
    if not os.path.exists(file_path):

        print('Creating %s'% file_path)
        
        os.makedirs(file_path)



        
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



ensure_dir(stan_cache)
