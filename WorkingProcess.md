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

