import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import input_manipulation
from RnnCrbm_class import rnncrbm

num = 3


num_conv_filters = 8
conv_strides = 2
span = 123
num_timesteps = 32

size_conv_filters = 4
hidden_width = int(np.floor((num_timesteps-size_conv_filters)/conv_strides) + 1)
n_hidden_recurrent = 100


def main(saved_weights_path):

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

    saver = tf.train.Saver(tvars)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        print(vars)
        for i in tqdm(range(num)):
            music = []
            music = sess.run(rcrbm.generate_music(music,10)) #Prime the network with song primer and generate an original song
            #print(music.shape)
            new_song_path = "./music_outputs/generated_{}".format(i) #The new song will be saved here
            input_manipulation.write_song(new_song_path, np.array(music))

if __name__ == "__main__":
    #main(int(sys.argv[1]))
    main('./parameter_checkpoints/epoch_14.ckpt')
