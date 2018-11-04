import h5py
import pandas as pd


def stanfit_to_hdf5(fit, file_name):
    """
    Write the parameters from a stan fit into a simple hdf5 file 
    
    :param fit: Stan fit object to save
    :param file_name: file to save the HDF5 data
    """
    extract = fit.extract()

    with h5py.File(file_name, 'w') as f:

        for key in extract.keys():

            f.create_dataset(key, data=extract[key], compression='lzf')


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

            for key in f.keys():

                v = f[key].value

                setattr(self, k, v)

                self._param_names.append(k)
                tmp = '('
                for n in v.shape:
                    tmp += '%d,'

                tmp[-1] = ')'

                self._param_dims.append(tmp)

    def display(self):
        """
        Display the properties of the stan fit parameters
        """

        out = {'parameter': self._param_names, 'dims': self._param_dims}

        df = pd.DataFrame(out)

        # put in display from ipython
        
        return df
 
