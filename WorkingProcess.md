##### Apr.24

Timeline:

* before 29th: poster (do in weekend)

##### Apr.19

* gibbs_sampling, the new prob is too high

##### Apr.17

Division of Work:

**Yutings**

* !! number & size of convolutional filters (by music nature)
* !! enforce sparsity to the crbm-visible-layer
* learning rate & epoch# for CRBM ('lr' & 'num_epochs' in *weight_intialization.py*  )
* learning rate for rnn-crbm: 'alpha = min(0.001, 0.0001/(float(i)))' in *rnn_rbm_train.py*
* batch_size  in *rnn_rbm_train.py*

#####Apr.7 

* feed our dataset and input into the rnn-rbm
* Problem:
  * why the matrix is too dense if we use timestep of 32
  * how to **constrain** that
  * probably we need to totally understand the network

****

* **Milestone:**
  * What we hv done:
    * dataset
    * input manipulation
    * got some results in rnn-rbm
    * (hvnt done yet) design the update of cnn
  * to do:
    * implement rnn cnn rbm
    * put some constraints in generation
    * smooth the music output

****





**Apr. 4**

* Finish Input Matrix generation

* Finish Note play Matrix to Midifile


**Apr. 3**

* main notes span:  (26 - 101 ) / (0 - 127)

  drum notes span: ( 35 - 81) / (0 - 127)

* resolution = ticks per Quarter note (480 for all)

* set 60 ticks to be one time step.  take round.



**March 30**

- Finally! Got Logic Pro! Midi edition problem solved! (Im so tired!)
- edit tons of game music midi file (45 songs actually). **Our dataset (main + drum track) is generated!**
- TODO:
  - Input manipulation, use RNN-RBM with our own input manipulation to generate
  - trying to find some pattern in our dataset (help decide **the window of RNN** & **convolution and strade** size)



**March 28 (2018)**

* Input manipulation (to do)
   * manually modify midi data file ( NES: https://www.vgmusic.com/music/console/nintendo/nes/ )
   * Note state matrix -> note play matrix ( main track & bass track )
   * figure out the proper span
* Theory
   * the weight update math
   * how to add cnn? 1D or 2D ? again cnn update (see CNN RBM paper)
   * boolean?????

