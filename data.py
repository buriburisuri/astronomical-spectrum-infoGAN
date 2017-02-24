import sugartensor as tf
import numpy as np


__author__ = 'namju.kim@kakaobrain.com'


class SpectrumData(object):

    def __init__(self, batch_size=128):

        # load data
        x = np.genfromtxt('asset/data/1000spectra.txt', delimiter=' ', dtype=np.float32)
        x = x[:, :-20:10]  # striding by step 10 ==> make 720 columns

        # normalize data between 0 and 1
        x /= 1.1  # max value is 1.1

        # make to Tensor
        X = x.reshape((x.shape[0], x.shape[1], 1, 1))

        # save to member variable
        self.batch_size = batch_size
        self.X = tf.sg_data._data_to_tensor([X], batch_size, name='train')
        self.num_batch = X.shape[0] // batch_size
