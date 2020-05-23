import stan_utility
import os
import numpy as np

def test_compile_file():
    model = stan_utility.compile_model(os.path.join(os.path.dirname(__file__), 'test.stan'))
    data = dict(
        mean=1,
        unused=np.random.normal(size=(4,42)),
    )
    stan_utility.sample_model(model, data, chains=2)

def test_compile_string():
    model_code = open(os.path.join(os.path.dirname(__file__), 'test.stan')).read()
    model = stan_utility.compile_model_code(model_code)
    data = dict(
        mean=1,
        unused=np.random.normal(size=(4,42)),
    )
    samples = stan_utility.sample_model(model, data, chains=1)
    stan_utility.plot_corner(samples)


if __name__ == '__main__':
    test_compile_string()
    test_compile_file()
