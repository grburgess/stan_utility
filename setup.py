#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Command
import os
import io
import sys
from shutil import rmtree


# Package meta-data.
NAME = "stan_utility"
DESCRIPTION = "Helper routines for pystan based off @betanalphas stan_utility"
URL = "https://github.com/grburgess/stan_utility"
EMAIL = "jmichaelburgess@gmail.com"
AUTHOR = "J. Michael Burgess"
REQUIRES_PYTHON = ">=2.7.0"
VERSION = None

REQUIRED = ["numpy", "scipy", "pystan<3", "joblib", "pandas", "arviz"]
TEST_REQUIRED = ["pytest>=3", "matplotlib", "getdist"]
SETUP_REQUIRED = ["pytest-runner", ]
# What packages are optional?
EXTRAS = {
    'plot': ['matplotlib', 'getdist'],
}


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution...")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        self.status("Pushing git tags...")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


stan_cache = os.path.expanduser("~/.stan_cache")


def ensure_dir(file_path):
    if not os.path.exists(file_path):

        print("Creating %s" % file_path)

        os.makedirs(file_path)


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    setup_requires=SETUP_REQUIRED,
    test_suite='tests',
    tests_require=TEST_REQUIRED,
    include_package_data=True,
    license="GPL",
    cmdclass={"upload": UploadCommand},
)


ensure_dir(stan_cache)
