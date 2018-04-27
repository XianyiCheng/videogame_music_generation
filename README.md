
This repository contains code video game music generation.

Our code is modified based on the rnn rbm code in https://github.com/dshieble/Music_RBM.



to train rnn-crbm:

python crbm_weight_initializations.py

python rcrbm_train.py <num_epochs>



to generate music:

python rcrbm_generate.py

(parameter_checkpoints/xxx.ckpt)



