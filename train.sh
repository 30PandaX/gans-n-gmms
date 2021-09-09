#!/bin/bash

source activate braindecoding

cd /home/zt246/storage/celeba/gans-n-gmms
python -u mfa_train_celeba.py --num_components 75 --samples_per_sub_component 10

