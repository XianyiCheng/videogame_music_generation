import tensorflow as tf
import numpy as np
import glob

from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

import CRBM
import input_manipulation

num_conv_filters = 8
conv_strides = 2
span = 123
num_timesteps = 32

size_conv_filters = 4
hidden_width = int(np.floor((num_timesteps-size_conv_filters)/conv_strides) + 1)
n_hidden_recurrent = 100

def rnncrbm():

    #x  = tf.placeholder(tf.float32, [None, num_timesteps, span], name="x")
    xt  = tf.placeholder(tf.float32, [num_timesteps, span], name="xt")

    lr  = tf.placeholder(tf.float32)
    batch_size = tf.placeholder(tf.int64, [1], name="batch_size")
    #parameters of CRBM
    W   = tf.Variable(tf.truncated_normal([size_conv_filters, span, 1, num_conv_filters], 0.001), name="W") #The weight matrix of the RBM
    bh  = tf.Variable(tf.zeros([hidden_width,num_conv_filters], tf.float32), name="bh") #The RNN -> RBM hidden bias vector
    bv  = tf.Variable(tf.zeros([num_timesteps, span], tf.float32), name="bv")#The RNN -> RBM visible bias vector

    #parameters related to RNN
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, int(hidden_width*num_conv_filters)], 0.0001), name="Wuh")  #The RNN -> RBM hidden weight matrix
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, int(num_timesteps*span)], 0.0001), name="Wuv") #The RNN -> RBM visible weight matrix
    Wvu = tf.Variable(tf.random_normal([int(num_timesteps*span), n_hidden_recurrent], 0.0001), name="Wvu") #The data -> RNN weight matrix
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001), name="Wuu") #The RNN hidden unit weight matrix
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32), name="bu")   #The RNN hidden unit bias vector
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent], tf.float32), name="u0") #The initial state of the RNN
    utm1 = tf.Variable(tf.zeros([1, n_hidden_recurrent], tf.float32), name="ut")

    #The RBM bias vectors. These matrices will get populated during rnn-rbm training and generation
    BH_t = tf.Variable(tf.ones([hidden_width,num_conv_filters],  tf.float32), name="BH_t")
    BV_t = tf.Variable(tf.ones([num_timesteps, span],  tf.float32), name="BV_t")

    def rnn_recurrence(u_tm1, v_t):
        # compute u_t
        v_t = tf.reshape(v_t, [1,num_timesteps*span])
        u_t = tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu))
        return u_t

    def visible_bias_recurrence(u_tm1):
        bv_t = tf.add(tf.reshape(bv,[1, int(num_timesteps*span)]), tf.matmul(u_tm1, Wuv))
        bv_t = tf.reshape(bv_t, [num_timesteps, span])
        return bv_t

    def hidden_bias_recurrence(u_tm1):
        bh_t = tf.add(tf.reshape(bh,[1, int(hidden_width*num_conv_filters)]), tf.matmul(u_tm1, Wuh))
        bh_t = tf.reshape(bh_t, [hidden_width,num_conv_filters])
        return bh_t

    def recurrence(k, u_tm1, x_t):
        bv_t = visible_bias_recurrence(u_tm1)
        bh_t = hidden_bias_recurrence(u_tm1)
        x_out = CRBM.gibbs_sample(x_t, W, bv_t, bh_t, k=5)
        u_t  = rnn_recurrence(u_tm1, x_out)
        tf.assign(utm1,u_t)
        cost = CRBM.free_energy_cost(x_t, x_out, W, bv_t, bh_t)
        return u_t, x_out, cost

    def generate_recurrence(count, k, u_tm1, primer, x, music):
        #This function builds and runs the gibbs steps for each RBM in the chain to generate music
        #Get the bias vectors from the current state of the RNN
        [u_t, x_out, _] = recurrence(k, u_tm1, primer)

        #Add the new output to the musical piece
        music = tf.concat([music, x_out], 0)
        return count+1, k, u_t, x_out, x, music


    [_, _, cost] = recurrence(5, utm1, xt)

    return xt, utm1, cost, W, bh, bv, lr, Wuh, Wuv, Wvu, Wuu, bu, u0
