#!/bin/bash

echo "Running pool $1"
cd /code/Diffusion-MU-Attack && conda init bash
echo "source activate concept" > ~/.bashrc && source ~/.bashrc
nohup python running_pool.py > /storage/malnick/concept_attacks/outputs/running_pool_$1.log 2> /storage/malnick/concept_attacks/outputs/running_pool_$1.err
