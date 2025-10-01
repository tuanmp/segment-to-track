#!/bin/bash
gpu_requirement="gpu&a100&hbm40"
# gpu_requirement="gpu"
salloc -A m3443_g -C $gpu_requirement -q interactive --nodes 1 --ntasks-per-node 1 --gpus-per-task 1 --cpus-per-task 32 --mem-per-gpu 32G --time 01:00:00 --gpu-bind=none --signal=SIGUSR1@180 #--image=tuanpham1503/torch_conda:0.4
