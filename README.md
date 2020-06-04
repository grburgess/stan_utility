# stan_utility
Utilities for PyStan for 

* caching model compilation in a smart way
* caching sampling results in a smart way
* checking Stan warnings
* making corner plots of scalar parameters

[![Build Status](https://travis-ci.org/JohannesBuchner/stan_utility.svg?branch=master)](https://travis-ci.org/JohannesBuchner/stan_utility)
[![PyPI version fury.io](https://badge.fury.io/py/stan-utility.svg)](https://pypi.python.org/pypi/stan_utility/)
[![PyPI license](https://img.shields.io/pypi/l/stan-utility.svg)](https://pypi.python.org/pypi/stan_utility/)

## Install 

	$ pip install stan-utility

## Usage

	import stan_utility

	model = stan_utility.compile_model('myscript.stan')
	data = dict(mean=1)
	
	results = stan_utility.sample_model(model, data, chains=2, iter=1000)
	print(results.posterior)  ## a arviz.InferenceData object
	
	# create mytest_fit_corner.pdf:
	stan_utility.plot_corner(samples, outprefix="mytest_fit")

On the second run of this code,

* compile_model will retrieve the compiled model from cache
* sample_model will retrieve the results from cache. (change the seed or parameters if you want a fresh run).

## Jupyter notebook features

See demos:

* https://github.com/JohannesBuchner/stan_utility/blob/master/examples/rosen2d.ipynb
* https://github.com/JohannesBuchner/stan_utility/blob/master/examples/rosenNd.ipynb

	import stan_utility

	model = stan_utility.compile_model_code("""
	data {
	   // maybe later
	}
	parameters {
		real<lower=0,upper=1> x;
		real y[10];
	}
	model {
		y ~ normal(1, 2);
	}
	""")
	data = dict(something=1)

	results = stan_utility.sample_model(model, data)
	print(results.posterior)
	
	stan_utility.plot_corner(samples)

Editing the comments, adding and removing lines will not require
recompilation of the model, because empty lines and comments are stripped out.

## Contributors

Derived originally from Stan_utility by Michael Betancourt and Sean Talts. 

* @grburgess
* @cescalara
* @JohannesBuchner

Contributions are welcome.

