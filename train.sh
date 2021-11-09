#!/bin/bash

source activate braindecoding

cd /home/px48/storage/celeba/gans-n-gmms
python -u mfa_train_celeba.py --num_components 100 --samples_per_sub_component 10 --output_dir ./concatenated_results_32

