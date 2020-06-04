import os
import glob
import joblib

path = os.path.expanduser("~/.stan_cache")
mem = joblib.Memory(path, verbose=False)

__all__ = ["mem", "get_path", "clear"]

def get_path():
    return path


def clear():
    # clear joblib memory
    mem.clear()
    # glob all the cache files
    files = glob.glob(os.path.join(get_path(), "cached*.pkl"))
    for f in files:
        os.remove(f)
