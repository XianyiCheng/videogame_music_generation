import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd
import pdb

"""
    This file contains the TF implementation of the convolutional Restricted Boltzman Machine
"""
num_conv_filters = 8
conv_strides = 2
span = 123
num_timesteps = 32

#This function lets us easily sample from a vector of probabilities
def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def crbm_inference(v, W, bh):
    conv_h = tf.nn.convolution(tf.reshape(v,[1,num_timesteps,span,1]),W,padding = 'VALID',strides = [conv_strides,1])
    prob_h = tf.add(tf.reshape(conv_h,bh.shape), bh)
    return tf.reshape(tf.sigmoid(prob_h),bh.shape)

def crbm_reconstruct(h,W,bv):
    deconv_v = tf.nn.conv2d_transpose(tf.reshape(h,[1,h.shape[0],1,h.shape[1]]),W,output_shape = [1,num_timesteps,span,1],strides = [1,conv_strides,1,1],padding = 'VALID')
    prob_v = tf.add(tf.reshape(deconv_v,[num_timesteps,span]), bv)
    return tf.reshape(tf.sigmoid(prob_v),[num_timesteps,span])

#This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(x, W, bv, bh, k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        prob_hk = crbm_inference(xk,W,bh)
        hk = sample(prob_hk) #Propagate the visible values to sample the hidden values
        prob_xk = crbm_reconstruct(hk,W,bv)
        xk = sample(prob_xk) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    cond = lambda count, k, x: tf.less(count,k)
    [_,_, x_sample] = tf.while_loop(cond, gibbs_step, [ct, k, x])
    #[_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter, gibbs_step, [ct, tf.constant(k), x], 1, False)

    #We need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def free_energy(x, h, W, bv, bh):
    return -tf.reduce_sum(tf.log(1 + tf.exp(h))) - tf.reduce_sum(tf.matmul(x, tf.transpose(bv)))

def get_free_energy_cost(x, W, bv, bh, k):
    #We use this function in training to get the free energy cost of the RBM. We can pass this cost directly into TensorFlow's optimizers
    #First, draw a sample from the RBM
    x_sample = gibbs_sample(x, W, bv, bh, k)

    def F(xx):
        #The function computes the free energy of a visible vector.
        hh = crbm_inference(xx,W,bh)
        return free_energy(xx, hh, W, bv, bh)

    #The cost is based on the difference in free energy between x and xsample
    cost = tf.reduce_mean(tf.subtract(F(x), F(x_sample)))
    return cost

def get_cd_update(x, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm.

    #First, we get the samples of x and h from the probability distribution
    #The sample of x
    x_sample = gibbs_sample(x, W, bv, bh, k)
    #The sample of the hidden nodes, starting from the visible state of x
    h = crbm_inference(x,W,bh)
    #The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = crbm_inference(x_sample, W, bh)

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    #W_  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    fc = free_energy(x, h, W, bv, bh) - free_energy(x_sample, h_sample, W, bv, bh)
    W_ = tf.multiply(-lr,tf.gradients(fc, W, stop_gradients = W))
    W_ = tf.reshape(W_,W.shape)
    bv_ = tf.multiply(-lr, tf.gradients(fc, bv, stop_gradients = bv))
    bv_ = tf.reshape(bv_, bv.shape)
    bh_ = tf.multiply(-lr, tf.gradients(fc, bh, stop_gradients = bh))
    bh_ = tf.reshape(bh_,bh.shape)
    #bv_ = tf.multiply(lr, tf.subtract(x, x_sample))
    #bh_ = tf.multiply(lr, tf.subtract(h, h_sample))

    return W_, bv_, bh_


def get_cd_update_batch(X, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm.
    #The batch size
    size_bt = tf.cast(tf.shape(X)[0],tf.float32)

    def cd_update_loop(W_, bv_, bh_, count):
        x = X[count,:,:]
        [W_c, bv_c, bh_c] = get_cd_update(x, W, bv, bh, k, lr/size_bt)
        W_ = tf.add(W_,W_c)
        bv_ = tf.add(bv_,bv_c)
        bh_ = tf.add(bh_,bh_c)
        return W_, bv_, bh_, count+1

    lr = tf.constant(lr, tf.float32) #The CD learning rate

    loop_vars = [tf.zeros(W.shape), tf.zeros(bv.shape), tf.zeros(bh.shape), tf.constant(0)]
    cond = lambda W_, bv_, bh_, count: tf.less(tf.cast(count,tf.float32),size_bt)
    [W_, bv_, bh_, _] = tf.while_loop(cond, cd_update_loop, loop_vars)

    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt
