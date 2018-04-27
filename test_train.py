import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import input_manipulation
from RnnCrbm_class import rnncrbm

"""
    This file contains the code for training the RNN-RBM by using the data in the Pop_Music_Midi directory
"""
num_conv_filters = 8
conv_strides = 2
span = 123
num_timesteps = 32

size_conv_filters = 4
hidden_width = int(np.floor((num_timesteps-size_conv_filters)/conv_strides) + 1)
n_hidden_recurrent = 100

batch_size = 100 #The number of trianing examples to feed into the rnn_rbm at a time
epochs_to_save = 5 #The number of epochs to run between saving each checkpoint
saved_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file

def main(num_epochs):
    W   = tf.Variable(tf.truncated_normal([size_conv_filters, span, 1, num_conv_filters], 0.001), name="W") #The weight matrix of the RBM
    bh  = tf.Variable(tf.zeros([hidden_width,num_conv_filters], tf.float32), name="bh") #The RNN -> RBM hidden bias vector
    bv  = tf.Variable(tf.zeros([num_timesteps, span], tf.float32), name="bv")#The RNN -> RBM visible bias vector
    #parameters related to RNN
    Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, int(hidden_width*num_conv_filters)], 0.0001), name="Wuh")  #The RNN -> RBM hidden weight matrix
    Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, int(num_timesteps*span)], 0.0001), name="Wuv") #The RNN -> RBM visible weight matrix
    Wvu = tf.Variable(tf.random_normal([int(num_timesteps*span), n_hidden_recurrent], 0.0001), name="Wvu") #The data -> RNN weight matrix
    Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001), name="Wuu") #The RNN hidden unit weight matrix
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32), name="bu")   #The RNN hidden unit bias vector
    utm1 = tf.Variable(tf.zeros([1, n_hidden_recurrent], tf.float32), name="ut")

    rcrbm = rnncrbm(W, bh, bv, Wuh, Wuv, Wvu, Wuu, bu, utm1)
    #The trainable variables include the weights and biases of the RNN and the RBM, as well as the initial state of the RNN
    tvars = [rcrbm.W, rcrbm.Wuh, rcrbm.Wuv, rcrbm.Wvu, rcrbm.Wuu, rcrbm.bh, rcrbm.bv, rcrbm.bu]
    cost = rcrbm.compute_cost()
    #The learning rate of the  optimizer is a parameter that we set on a schedule during training
    opt_func = tf.train.GradientDescentOptimizer(learning_rate=rcrbm.lr)
    gvs = opt_func.compute_gradients(cost, tvars)
    def ClipIfNotNone(grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -10., 10.)

    gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
    updt = opt_func.apply_gradients(gvs)#The update step involves applying the clipped gradients to the model parameters

    songs = input_manipulation.get_songs('Game_Music_Midi') #Load the songs

    saver = tf.train.Saver(tvars) #We use this saver object to restore the weights of the model and save the weights every few epochs
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        #We run through all of the songs n_epoch times
        print ("starting")
        for epoch in range(num_epochs):
            costs = []
            start = time.time()
            for s_ind, song in enumerate(songs):
                for i in range(song.shape[0]):
                    tr_x = song[i,:,:]
                    alpha = 0.0001 #We decrease the learning rate according to a schedule.
                    _, C = sess.run([updt, cost], feed_dict={rcrbm.xt: tr_x, rcrbm.lr: alpha})
                    costs.append(C)
            #Print the progress at epoch
            print ("epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start))
            print
            #Here we save the weights of the model every few epochs
            if (epoch + 1) % epochs_to_save == 0:
                saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))
