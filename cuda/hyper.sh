#!/bin/bash

#SBATCH --job-name=hypergraphsMatching
#SBATCH --output=res_hypergraphsMatching
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export CUDA_VISIBLE_DEVICES=1

./hyper.out test-images/monster1m.JPG test-images/monster1m.rot.JPG
