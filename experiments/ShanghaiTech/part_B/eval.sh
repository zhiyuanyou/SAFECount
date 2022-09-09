#!/bin/bash
ROOT=../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

g=$(($1<8?$1:8))
spring.submit run \
    -p$2 \
    -n$g \
    --gres=gpu:$g \
    --ntasks-per-node=$g \
    --cpus-per-task 4 \
    --job-name=partb \
    --gpu \
    "python -u ../../../tools/train_val.py -e"
