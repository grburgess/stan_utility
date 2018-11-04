import h5py
import pandas as pd


def stanfit_to_hdf5(fit, file_name):
    """
    Write the parameters from a stan fit into a simple hdf5 file 
    
    :param fit: Stan fit object to save
    :param file_name: file to save the HDF5 data
    """
    extract = fit.extract(permuted=False, inc_warmup=False)

    with h5py.File(file_name, 'w') as f:

        params_grp = f.create_group('parameters')

        for key in extract.keys():

            params_grp.create_dataset(key, data=extract[key], compression='lzf')

        sampler_grp = f.create_group('sampler')

        for k, v in fit.get_sampler_params(inc_warmup=False).items():

            new_key = k.replace('__', '')

            sampler_grp.create_dataset(new_key, data=v, compression='lzf')


class StanSavedFit(object):

    def __init__(self, file_name):
        """
        Load a Stan fit from an HDF5 file created by stanfit_to_hdf5.
        The parameters are accessed as class attributes.
        :param file_name: The file name of the saved stan fit
        """

        self._param_names = []
        self._param_dims = []

        with h5py.File(file_name, 'r') as f:

            # attach the parameters
            # as members of the class
            
            p = f['parameters']

            for key in p.keys():

                v = f[key].value

                setattr(self, k, v)

                self._param_names.append(k)
                tmp = '('
                for n in v.shape:
                    tmp += '%d,'

                tmp[-1] = ')'

                self._param_dims.append(tmp)

        self._file_name = file_name


    # @property
    # def divergent_transitions(self):
        
    #     with h5py.File(self._file_name, 'r') as f:

            
        

    def display(self):
        """
        Display the properties of the stan fit parameters
        """

        out = {'parameter': self._param_names, 'dims': self._param_dims}

        df = pd.DataFrame(out)

        # put in display from ipython

        return df
