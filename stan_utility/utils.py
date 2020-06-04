import pickle
import numpy
import os
import hashlib
import re
import warnings
import collections

import pystan
import arviz

from stan_utility.cache import get_path as get_path_of_cache


def check_div(fit, quiet=False):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y["divergent__"]]
    n = sum(divergent)
    N = len(divergent)

    if not quiet and n > 0:
        warnings.warn(
            "{} of {} iterations ended with a divergence ({}%)".format(
                n, N, 100 * n / N
            )
        )

    if n > 0:
        if not quiet:
            warnings.warn("  Try running with larger adapt_delta to remove the divergences")
        else:
            return False
    else:
        if quiet:
            return True


def check_treedepth(fit, max_treedepth=10, quiet=False):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y["treedepth__"]]
    n = sum(1 for x in depths if x == max_treedepth)
    N = len(depths)

    if not quiet and n > 0:
        warnings.warn(
            (
                "{} of {} iterations saturated the maximum tree depth of {}" + " ({}%)"
            ).format(n, N, max_treedepth, 100 * n / N)
        )
    if n > 0:
        if not quiet:
            warnings.warn(
                "  Run again with max_treedepth set to a larger value to avoid saturation"
            )
        else:
            return False
    else:
        if quiet:
            return True


def check_energy(fit, quiet=False):
    """Checks the energy fraction of missing information (E-FMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    no_warning = True
    for chain_num, s in enumerate(sampler_params):
        energies = s["energy__"]
        numer = sum(
            (energies[i] - energies[i - 1]) ** 2 for i in range(1, len(energies))
        ) / len(energies)
        denom = numpy.var(energies)
        if numer / denom < 0.2:
            if not quiet:
                warnings.warn("Chain {}: E-BFMI = {}".format(chain_num, numer / denom))
            no_warning = False

    if no_warning:
        if quiet:
            return True
    else:
        if not quiet:
            warnings.warn(
                "  E-BFMI below 0.2 indicates you may need to reparameterize your model"
            )
        else:
            return False


def check_n_eff(fit, quiet=False):
    """Checks the effective sample size per iteration"""
    fit_summary = fit.summary(probs=[0.5])
    n_effs = [x[4] for x in fit_summary["summary"]]
    names = fit_summary["summary_rownames"]
    n_iter = len(fit.extract()["lp__"])

    no_warning = True
    for n_eff, name in zip(n_effs, names):
        ratio = n_eff / n_iter
        if ratio < 0.001:
            if not quiet:
                warnings.warn("n_eff / iter for parameter {} is {}!".format(name, ratio))
            no_warning = False

    if no_warning:
        if quiet:
            return True
    else:
        if not quiet:
            warnings.warn(
                "  n_eff / iter below 0.001 indicates that the effective sample size has likely been overestimated"
            )
        else:
            return False


def check_rhat(fit, quiet=False):
    """Checks the potential scale reduction factors"""
    from math import isnan
    from math import isinf

    fit_summary = fit.summary(probs=[0.5])
    rhats = [x[5] for x in fit_summary["summary"]]
    names = fit_summary["summary_rownames"]

    no_warning = True
    for rhat, name in zip(rhats, names):
        if rhat > 1.1 or isnan(rhat) or isinf(rhat):
            if not quiet:
                warnings.warn("Rhat for parameter {} is {}!".format(name, rhat))
            no_warning = False
    if no_warning:
        if quiet:
            return True
    else:
        if not quiet:
            warnings.warn(
                "  Rhat above 1.1 indicates that the chains very likely have not mixed"
            )
        else:
            return False


def check_all_diagnostics(fit, max_treedepth=10, quiet=False):
    """Checks all MCMC diagnostics"""

    if not quiet:
        check_n_eff(fit)
        check_rhat(fit)
        check_div(fit)
        check_treedepth(fit, max_treedepth=max_treedepth)
        check_energy(fit)
    else:
        warning_code = 0
        if not check_n_eff(fit, quiet):
            warning_code = warning_code | (1 << 0)
        if not check_rhat(fit, quiet):
            warning_code = warning_code | (1 << 1)
        if not check_div(fit, quiet):
            warning_code = warning_code | (1 << 2)
        if not check_treedepth(fit, max_treedepth, quiet):
            warning_code = warning_code | (1 << 3)
        if not check_energy(fit, quiet):
            warning_code = warning_code | (1 << 4)

        return warning_code


def parse_warning_code(warning_code):
    """Parses warning code into individual failures"""
    if warning_code & (1 << 0):
        warnings.warn("n_eff / iteration warning")
    if warning_code & (1 << 1):
        warnings.warn("rhat warning")
    if warning_code & (1 << 2):
        warnings.warn("divergence warning")
    if warning_code & (1 << 3):
        warnings.warn("treedepth warning")
    if warning_code & (1 << 4):
        warnings.warn("energy warning")


def _by_chain(unpermuted_extraction):
    num_chains = len(unpermuted_extraction[0])
    result = [[] for _ in range(num_chains)]
    for c in range(num_chains):
        for i in range(len(unpermuted_extraction)):
            result[c].append(unpermuted_extraction[i][c])
    return numpy.array(result)


def _shaped_ordered_params(fit):
    ef = fit.extract(
        permuted=False, inc_warmup=False
    )  # flattened, unpermuted, by (iteration, chain)
    ef = _by_chain(ef)
    ef = ef.reshape(-1, len(ef[0][0]))
    ef = ef[:, 0 : len(fit.flatnames)]  # drop lp__
    shaped = {}
    idx = 0
    for dim, param_name in zip(fit.par_dims, fit.extract().keys()):
        length = int(numpy.prod(dim))
        shaped[param_name] = ef[:, idx : idx + length]
        shaped[param_name].reshape(*([-1] + dim))
        idx += length
    return shaped


def partition_div(fit):
    """ Returns parameter arrays separated into divergent and non-divergent transitions"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    div = numpy.concatenate([x["divergent__"] for x in sampler_params]).astype("int")
    params = _shaped_ordered_params(fit)
    nondiv_params = dict((key, params[key][div == 0]) for key in params)
    div_params = dict((key, params[key][div == 1]) for key in params)
    return nondiv_params, div_params

def trim_model_code(code):
    """Strip white space, empty lines and comments from stan code."""
    lines = code.split("\n")
    lines = [re.sub('//.*$', '', line).strip() for line in lines]
    lines = [line.replace('    ', ' ').replace('  ', ' ').replace('  ', ' ')
        for line in lines if len(line) > 0]

    slimcode = '\n'.join(lines).strip()
    slimbytes = slimcode.encode(encoding="ascii", errors="ignore")
    slimcode = slimbytes.decode(encoding="ascii")
    return slimcode

def hash_model_code(code):
    """Get a hash for stan code"""
    slimbytes = code.encode(encoding="ascii")
    hexcode = hashlib.md5(slimbytes).hexdigest()
    return hexcode

def compile_model(filename, model_name="anon_model", print_code=False, **kwargs):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    if print_code is True, gives line numbers of code compiled.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""

    with open(filename) as f:
        model_code = f.read()
    return compile_model_code(model_code, model_name=model_name, print_code=print_code, simplify=False, **kwargs)

def compile_model_code(model_code, model_name="anon_model", print_code=True, simplify=True, **kwargs):
    """This will automatically cache models - great if you're just running a
    script on the command line.

    If print_code is True, gives line numbers of code compiled. Useful to quickly
    find the line in error. If simplify is True, strips empty lines and comments from code.

    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""

    simple_model_code = trim_model_code(model_code)
    code_hash = hash_model_code(simple_model_code)
    if model_name is None:
        cache_fn = os.path.join(
            get_path_of_cache(), "cached-model-{}.pkl".format(code_hash)
        )
    else:
        cache_fn = os.path.join(
            get_path_of_cache(), "cached-{}-{}.pkl".format(model_name, code_hash)
        )

    try:
        sm = pickle.load(open(cache_fn, "rb"))
        print("Using cached StanModel")
        return sm
    except IOError:
        pass
    
    if simplify:
        model_code = simple_model_code

    if print_code:
        formatted_code = '\n'.join(['%3d: %s' % (i+1, line)
            for i, line in enumerate(model_code.split("\n"))])
        print("Compiling slimmed Stan code[%s_%s]:\n%s" % (model_name, code_hash, formatted_code))

    if model_name is not None:
        kwargs['model_name'] = model_name

    sm = pystan.StanModel(model_code=model_code, **kwargs)

    with open(cache_fn, "wb") as f:
        pickle.dump(sm, f)

    return sm


def fast_extract(fit, spec):
    """
    Faster extraction of generated quantities from Stan than using StanFit4Model.extract().
    Unfortunately it requires specifying the order and shape of *all* params.
    
    Based on discussion/temporary solution here:
    https://github.com/stan-dev/pystan/issues/462

    :param fit: pystan fit object to extract from.
    :param spec: a dict of parameter keys and shape pairs
    e.g. spec = {'alpha' : 10, 'beta' : (100, 100), 'gamma' : (10, 100, 200)} etc...

    :returns: dict of params acessed by key (as with StanFit4Model.extract())
    """

    all_output = numpy.array(
        [float(i) for i in fit.sim["samples"][0].chains.values()][:-1]
    )

    organised_output = {}
    index = 0

    # run through spec
    for key, shape in spec.items():

        # take the required chunk and reshape
        n = numpy.prod(shape)
        tmp = all_output[index : index + n]
        tmp = tmp.reshape(shape)

        # assign to key
        organised_output[key] = tmp
        index += n

    return organised_output


from stan_utility.cache import mem as cache_mem
@cache_mem.cache(ignore=["refresh"])
def _sample_model(model, data, refresh=100, **kwargs):
    print()
    print("Data")
    print("----")
    for k, v in data.items():
        if numpy.shape(v) == ():
            print('  %-10s: %s' % (k, v))
        else:
            print('  %-10s: shape %s [%s ... %s]' % (k, numpy.shape(v), numpy.min(v), numpy.max(v)))
    
    print()
    print("sampling from model ...")
    fit = model.sampling(data=data, refresh=refresh, **kwargs)
    print("processing results ...")
    print(fit)
    print("checking results ...")
    check_all_diagnostics(fit,
        max_treedepth=kwargs.get('control', {}).get('max_treedepth', 10),
        quiet=False)
    return arviz.convert_to_inference_data(fit)

def sample_model(model, data, outprefix=None, **kwargs):
    """
    Sample Stan model and write the parameters into a simple hdf5 file

    :param model: Stan model to sample from
    :param data: data to pass to model
    :param file_name: HDF5 file name where samples will be stored

    All other arguments are passed to model.sampling().
    Result is cached.
    """
    fit = _sample_model(model, data, **kwargs)

    if outprefix is not None:
        arviz.to_netcdf(fit, outprefix + 'fit.hdf5')
    
    return fit

def get_flat_posterior(results):
    la = results.posterior.data_vars
    flat_posterior = collections.OrderedDict()
    for k, v in la.items():
        a = v.data
        newshape = tuple([a.shape[0] * a.shape[1]] + list(a.shape)[2:])
        flat_posterior[k] = v.data.reshape(newshape)
    return flat_posterior

def plot_corner(results, outprefix=None, **kwargs):
    """
    Store a simple corner plot in outprefix_corner.pdf, based on samples
    extracted from fit.

    Additional kwargs are passed to MCSamples.
    """
    la = get_flat_posterior(results)
    samples = []
    paramnames = []
    badlist = ['lp__']
    badlist += [k for k in la.keys() if 'log' in k and k.replace('log', '') in la.keys()]

    for k in sorted(la.keys()):
        print('%20s: %.4f +- %.4f' % (k, la[k].mean(), la[k].std()))
        if k not in badlist and la[k].ndim == 2:
            samples.append(la[k])
            paramnames.append(k)

    if len(samples) == 0:
        arrays = [k for k in la.keys() if la[k].ndim == 3 and la[k].shape[2] <= 20 and k not in badlist]
        if len(arrays) != 1:
            warnings.warn("no scalar variables found")
            return

        k = arrays[0]
        # flatten across chains and column for each variable
        samples = la[k]
        paramnames = ['%s[%d]' % (k, i + 1) for i in range(la[k].shape[1])]

    samples = numpy.transpose(samples)
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots
    settings = kwargs.pop('settings', dict(smooth_scale_2D=3.0))
    samples_g = MCSamples(samples=samples, names=paramnames, settings=settings, **kwargs)
    g = plots.get_subplot_plotter()
    g.settings.num_plot_contours = 3
    g.triangle_plot([samples_g], filled=False, contour_colors=plt.cm.Set1.colors);
    if outprefix is not None:
        plt.savefig(outprefix + '_corner.pdf', bbox_inches='tight')
        plt.close()

