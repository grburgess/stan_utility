import os
import glob

stan_cache = os.path.expanduser("~/.stan_cache")


def get_path_of_cache():

    return stan_cache


def clear_stan_cache(self):

    # glob all the cache files

    files = glob.glob(os.path.join(stan_cache, "cached*.pkl"))

    for f in files:

        os.remove(f)
