# stan_utility
Utilities for PyStan for 
* caching model compilation
* caching sampling results
* storing sampling results
* checking Stan warnings
* making corner plots of scalar parameters

## Install 

	$ pip install stan_utility

## Usage

	import stan_utility

	model = stan_utility.compile_model('myscript.stan')
	data = dict(mean=1)
	
	results = stan_utility.sample_model(model, data, chains=2, iter=1000)
	print(results["parameter"].std())
	
	stan_utility.plot_corner(samples, outprefix="mytest_fit")
	# mytest_fit_corner.pdf will be created

On the second run of this code
* compile_model will retrieve the compiled model from cache
* sample_model will retrieve the results from cache. (change the seed or parameters if you want a fresh run).

Usage for experimenting in a notebook:

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
	print(results["parameter"].std())
	
	stan_utility.plot_corner(samples)

Editing the comments, adding and removing lines will not require
recompilation of the model.


