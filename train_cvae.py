import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cf_vae_cpmf import cf_vae, params


np.random.seed(0)
tf.set_random_seed(0)

def load_cvae_data():
  data = {}
  data_dir = "data/citeulike-a/"
  variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  data["content"] = variables['X']

  data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items.dat")

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

params = params()
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1


# # for updating W and b in vae
# self.learning_rate = 0.001
# self.batch_size = 500
# self.num_iter = 3000
# self.EM_iter = 100


data = load_cvae_data()
num_factors = 50
model = cf_vae(num_users=5551, num_items=16980, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000],
    loss_type='cross_entropy')
model.fit(data["train_users"], data["train_items"], data["content"], params)
model.save_model("cf_vae.mat")
# model.load_model("cf_vae.mat")
recalls = model.predict(data['train_users'], data['test_users'], 350)

plt.figure()
plt.plot(np.arange(50, 350, 50),recalls)
plt.show()
