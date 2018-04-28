import numpy as np

span = 123
num_timesteps = 32

num_conv_filters = 8
size_conv_filters = 4
conv_strides = 2

hidden_width = int(np.floor((num_timesteps-size_conv_filters)/conv_strides) + 1)
n_hidden_recurrent = 100

# crbm weight initializations
crbm_num_epochs = 100
crbm_lr = 0.0001
