import os
import numpy as np
import tempfile
import joblib

def test_compile_file():
    import stan_utility.cache
    with tempfile.TemporaryDirectory() as cachedir:
        print("using cachedir:", cachedir)
        stan_utility.cache.path = cachedir
        stan_utility.cache.mem = joblib.Memory(cachedir, verbose=False)
        
        import stan_utility

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
    import stan_utility.cache
    with tempfile.TemporaryDirectory() as cachedir:
        print("using cachedir:", cachedir)
        stan_utility.cache.path = cachedir
        stan_utility.cache.mem = joblib.Memory(cachedir, verbose=False)
        
        import stan_utility

        model_code = open(os.path.join(os.path.dirname(__file__), 'test.stan')).read()
        model = stan_utility.compile_model_code(model_code, model_name="mytest")
        data = dict(
            mean=1,
            unused=np.random.normal(size=(4,42)),
        )
        if os.path.exists("mytest_fitfit.hdf5"):
            os.unlink("mytest_fitfit.hdf5")
        samples = stan_utility.sample_model(model, data, outprefix="mytest_fit", chains=2, iter=346)
        assert os.path.exists("mytest_fitfit.hdf5")
        os.unlink("mytest_fitfit.hdf5")

        if os.path.exists("mytest_fit_corner.pdf"):
            os.unlink("mytest_fit_corner.pdf")
        stan_utility.plot_corner(samples, outprefix="mytest_fit")
        assert os.path.exists("mytest_fit_corner.pdf")
        os.unlink("mytest_fit_corner.pdf")
        
        flat_samples = stan_utility.get_flat_posterior(samples)
        assert set(flat_samples.keys()) == {"x", "y"}, flat_samples.keys()
        assert flat_samples['x'].shape == (346,), flat_samples['x'].shape
        assert flat_samples['y'].shape == (346, 10), flat_samples['y'].shape



if __name__ == '__main__':
    test_compile_string()
    test_compile_file()
