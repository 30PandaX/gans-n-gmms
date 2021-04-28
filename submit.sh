#!/bin/bash

mkdir -p job_err
mkdir -p job_out

sbatch --requeue -p sablab -t 8:00:00 --mem=8G --gres=gpu:1 --job-name=$1 -e ./job_err/%j-$1.err -o ./job_out/%j-$1.out $2

