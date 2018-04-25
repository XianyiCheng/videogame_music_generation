import tensorflow as tf
import numpy as np
import input_manipulation as im

midifile = './Game_Music_Midi/1943sab_m.mid'

NotePlayMatrix = im.midiToNotePlayMatrix(midifile)
inputM = NotePlayMatrix[:32,:]
inputM = inputM.astype(np.float32)
inputM = np.reshape(inputM,(1,32,123,1))

conv_filters =  tf.Variable(tf.nn.l2_normalize(tf.random_normal([4, 123,1,8], 0.01),axis = [0,1]))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
data = tf.nn.l2_normalize(tf.random_normal([1,32, 123,1], 0.01),axis=1)
##print(sess.run(data))
#print('haha')
conv1 = tf.nn.convolution(data,conv_filters,padding = 'VALID',strides = [2,1])

h = sess.run(conv1)
print(h.shape
#h = h.reshape((15,8))
dc = tf.nn.conv2d_transpose(h,conv_filters,[1,32,123,1],[1,2,1,1],padding = 'VALID')
#print(sess.run(tf.nn.l2_normalize((sess.run(dc)),axis=[1])))
