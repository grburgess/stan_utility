import stan_utility
import os
import numpy as np
import matplotlib.pyplot as plt

def test_compile_file():
    model = stan_utility.compile_model(os.path.join(os.path.dirname(__file__), 'test.stan'))
    data = dict(
        mean=1,
        unused=np.random.normal(size=(4,42)),
    )
    stan_utility.sample_model(model, data, chains=2)

    files = os.listdir(stan_utility.cache.get_path())
    assert "joblib" in files
    assert any(f for f in files if f.startswith("cached-") and f.endswith('.pkl')), files
    assert len(files) > 1, files

    stan_utility.cache.clear()

    files = os.listdir(stan_utility.cache.get_path())
    assert files == ["joblib"], files


def test_compile_string():
    model_code = open(os.path.join(os.path.dirname(__file__), 'test.stan')).read()
    model = stan_utility.compile_model_code(model_code, model_name="mytest")
    data = dict(
        mean=1,
        unused=np.random.normal(size=(4,42)),
    )
    if os.path.exists("mytest_fitfit.hdf5"):
        os.unlink("mytest_fitfit.hdf5")
    samples = stan_utility.sample_model(model, data, outprefix="mytest_fit", chains=1)
    assert os.path.exists("mytest_fitfit.hdf5")
    os.unlink("mytest_fitfit.hdf5")

    if os.path.exists("mytest_fit_corner.pdf"):
        os.unlink("mytest_fit_corner.pdf")
    stan_utility.plot_corner(samples, outprefix="mytest_fit")
    assert os.path.exists("mytest_fit_corner.pdf")
    os.unlink("mytest_fit_corner.pdf")


if __name__ == '__main__':
    test_compile_string()
    test_compile_file()
