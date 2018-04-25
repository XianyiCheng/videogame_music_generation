import tensorflow as tf
import numpy as np
from tqdm import tqdm
import CRBM
import input_manipulation

"""
	This file stores the code for initializing the weights of the RNN-RBM. We initialize the parameters of the RBMs by
	training them directly on the data with CD-k. We initialize the parameters of the RNN with small weights.
"""

num_epochs = 100 #The number of epochs to train the CRBM
lr = 0.0001 #The learning rate for the CRBM


num_conv_filters = 8
conv_strides = 2
span = 123
num_timesteps = 32

size_conv_filters = 4
hidden_width = np.floor((num_timesteps-size_conv_filters)/conv_strides) + 1
n_hidden_recurrent = 100
def main():
	#Load the Songs
	tf.reset_default_graph()
	songs = input_manipulation.get_songs('Game_Music_Midi')
	x  = tf.placeholder(tf.float32, [None, num_timesteps, span], name="x")

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

	#The RBM bias vectors. These matrices will get populated during rnn-rbm training and generation
	BH_t = tf.Variable(tf.ones([hidden_width,num_conv_filters],  tf.float32), name="BH_t")
	BV_t = tf.Variable(tf.ones([num_timesteps, span],  tf.float32), name="BV_t")

	#Build the RBM optimization
	saver = tf.train.Saver()
	#Note that we initialize the RNN->RBM bias vectors with the bias vectors of the trained RBM. These vectors will form the templates for the bv_t and
	#bh_t of each RBM that we create when we run the RNN-RBM
	updt = CRBM.get_cd_update_batch(x, W, bv, bh, 1, lr)

	#Run the session
	with tf.Session() as sess:
		#Initialize the variables of the model
		init = tf.global_variables_initializer()
		sess.run(init)

	    #Run over each song num_epoch times
		for epoch in tqdm(range(num_epochs)):
			for song in songs:
				sess.run(updt, feed_dict={x: song})

	    #Save the initialized model here
		save_path = saver.save(sess, "parameter_checkpoints/initialized.ckpt")

if __name__ == "__main__":
    main()
