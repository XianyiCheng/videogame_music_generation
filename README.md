
This repository contains code video game music generation.

Our code is modified based on the rnn rbm code in https://github.com/dshieble/Music_RBM.



to train rnn-rbm:

python weight_initialization.py

python rnn_rbm_train.py <num_epochs>



to generate music:

python rnn_rbm_generate.py <path_to_ckpt_file>

(parameter_checkpoints/xxx.ckpt)



