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

def rnn_recurrence(u_tm1, v_t, Wvu, Wuu, bu):
    # compute u_t
    v_t = tf.reshape(v_t, [1,num_timesteps*span])
    u_t = tf.tanh(bu + tf.matmul(v_t, Wvu) + tf.matmul(u_tm1, Wuu))
    return u_t

def visible_bias_recurrence(u_tm1, Wuv, bv):
    bv_t = tf.add(tf.reshape(bv,[1, int(num_timesteps*span)]), tf.matmul(u_tm1, Wuv))
    bv_t = tf.reshape(bv_t, [num_timesteps, span])
    return bv_t

def hidden_bias_recurrence(u_tm1, Wuh, bh):
    bh_t = tf.add(tf.reshape(bh,[1, int(hidden_width*num_conv_filters)]), tf.matmul(u_tm1, Wuh))
    bh_t = tf.reshape(bh_t, [hidden_width,num_conv_filters])
    return bh_t

def recurrence(k, u_tm1, x_t, W, bh, bv, Wuh, Wuv, Wvu, Wuu, bu):
    bv_t = visible_bias_recurrence(u_tm1, Wuv, bv)
    bh_t = hidden_bias_recurrence(u_tm1, Wuh, bh)
    x_out = CRBM.gibbs_sample(x_t, W, bv_t, bh_t, k=5)
    u_t  = rnn_recurrence(u_tm1, x_out, Wvu, Wuu, bu)
    cost = CRBM.free_energy_cost(x_t, x_out, W, bv_t, bh_t)
    return u_t, x_out, cost

class rnncrbm:
    k = 1
    def __init__(self, W, bh, bv, Wuh, Wuv, Wvu, Wuu, bu, utm1):
        self.xt  = tf.placeholder(tf.float32, [num_timesteps, span], name="xt")
        self.lr  = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int64, [1], name="batch_size")
        self.W = W
        self.bh = bh
        self.bv = bv
        self.Wuh = Wuh
        self.Wuv = Wuv
        self.Wvu = Wvu
        self.Wuu = Wuu
        self.bu = bu
        self.utm1 = utm1

    def compute_cost(self):
        self.utm1, _, self.cost = recurrence(self.k, self.utm1, self.xt, self.W, self.bh, self.bv, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu)
        return self.cost

    def generate_music(self, music, T):
        vt = tf.zeros([num_timesteps, span], tf.float32)
        for i in range(T):
            [self.utm1, vt, _] = recurrence(self.k, self.utm1, vt, self.W, self.bh, self.bv, self.Wuh, self.Wuv, self.Wvu, self.Wuu, self.bu)
            vt = tf.reshape(vt,[num_timesteps, span])
            music.append(vt)
        return music
