import tensorflow as tf
import numpy as np
from vae import vanilla_vae
import scipy.io as sio

np.random.seed(0)
tf.set_random_seed(0)

variables = sio.loadmat("data/citeulike-a/mult_nor.mat")
data = variables['X']
idx = np.random.rand(data.shape[0]) < 0.8
train_X = data[idx]
test_X = data[~idx]

model = vanilla_vae(input_dim=8000, encoding_dims=[200, 100], z_dim=50, decoding_dims=[100, 200, 8000], loss='cross_entropy')
# As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                    # lr=0.01, batch_size=128, print_step=50)
print('fitting data starts...')
model.fit(train_X, epochs=10000,learning_rate=0.001, batch_size=500, print_size=50, train=True)
